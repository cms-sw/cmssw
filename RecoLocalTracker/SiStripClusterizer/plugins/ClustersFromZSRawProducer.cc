/*
 */
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingAlgorithms.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>
#include <memory>
#include <atomic>
#include <mutex>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

// #define VIDEBUG
#ifdef VIDEBUG
#include <iostream>
#define COUT std::cout << "VI "
#else
#define COUT LogDebug("")
#endif

namespace {
  std::unique_ptr<sistrip::FEDBuffer> fillBuffer(int fedId, const FEDRawDataCollection& rawColl) {
    std::unique_ptr<sistrip::FEDBuffer> buffer;

    // Retrieve FED raw data for given FED
    const FEDRawData& rawData = rawColl.FEDData(fedId);

    // Check on FEDRawData pointer
    if
      UNLIKELY(!rawData.data()) {
        if (edm::isDebugEnabled()) {
          edm::LogWarning(sistrip::mlRawToCluster_) << "[ClustersFromZSRawProducer::" << __func__ << "]"
                                                    << " NULL pointer to FEDRawData for FED id " << fedId;
        }
        return buffer;
      }

    // Check on FEDRawData size
    if
      UNLIKELY(!rawData.size()) {
        if (edm::isDebugEnabled()) {
          edm::LogWarning(sistrip::mlRawToCluster_) << "[ClustersFromZSRawProducer::" << __func__ << "]"
                                                    << " FEDRawData has zero size for FED id " << fedId;
        }
        return buffer;
      }

    // construct FEDBuffer
    try {
      buffer.reset(new sistrip::FEDBuffer(rawData.data(), rawData.size()));
      if
        UNLIKELY(!buffer->doChecks(false))
      throw cms::Exception("FEDBuffer") << "FED Buffer check fails for FED ID" << fedId << ".";
    } catch (const cms::Exception& e) {
      if (edm::isDebugEnabled()) {
        edm::LogWarning(sistrip::mlRawToCluster_)
            << "Exception caught when creating FEDBuffer object for FED " << fedId << ": " << e.what();
      }
      return std::unique_ptr<sistrip::FEDBuffer>();
    }

    /*
    // dump of FEDRawData to stdout
    if ( dump_ ) {
    std::stringstream ss;
    RawToDigiUnpacker::dumpRawData( fedId, rawData, ss );
    LogTrace(mlRawToDigi_) 
    << ss.str();
    }
    */

    return buffer;
  }

  class ClusterFiller {
  public:
    ClusterFiller(const FEDRawDataCollection& irawColl,
                  StripClusterizerAlgorithm& iclusterizer,
                  SiStripRawProcessingAlgorithms& irawAlgos,
                  bool idoAPVEmulatorCheck,
                  bool legacy,
                  bool hybridZeroSuppressed)
        : rawColl(irawColl),
          clusterizer(iclusterizer),
          rawAlgos(irawAlgos),
          doAPVEmulatorCheck(idoAPVEmulatorCheck),
          legacy_(legacy),
          hybridZeroSuppressed_(hybridZeroSuppressed) {
      incTot(clusterizer.allDets().size());
    }

    ~ClusterFiller() { printStat(); }

    void fill(StripClusterizerAlgorithm::Det const & det, StripClusterizerAlgorithm::output_t::FastFiller& record);

  private:
    std::unique_ptr<sistrip::FEDBuffer> buffers[1024];

    const FEDRawDataCollection& rawColl;

    StripClusterizerAlgorithm& clusterizer;
    SiStripRawProcessingAlgorithms& rawAlgos;

    // March 2012: add flag for disabling APVe check in configuration
    bool doAPVEmulatorCheck;

    bool legacy_;
    bool hybridZeroSuppressed_;
// #define VISTAT
#ifdef VISTAT
    struct Stat {
      Stat() : totDet(0), detReady(0), detSet(0), detAct(0), detNoZ(0), detAbrt(0), totClus(0) {}
      std::atomic<int> totDet;    // all dets
      std::atomic<int> detReady;  // dets "updated"
      std::atomic<int> detSet;    // det actually set not empty
      std::atomic<int> detAct;    // det actually set with content
      std::atomic<int> detNoZ;    // det actually set with content
      std::atomic<int> detAbrt;   // det aborted
      std::atomic<int> totClus;   // total number of clusters
    };

    mutable Stat stat;
    // void zeroStat() const { stat = std::move(Stat()); }
    void incTot(int n) const { stat.totDet = n; }
    void incReady() const { stat.detReady++; }
    void incSet() const { stat.detSet++; }
    void incAct() const { stat.detAct++; }
    void incNoZ() const { stat.detNoZ++; }
    void incAbrt() const { stat.detAbrt++; }
    void incClus(int n) const { stat.totClus += n; }
    void printStat() const {
      std::cout << "VI clusters " << stat.totDet << ',' << stat.detReady << ',' << stat.detSet << ',' << stat.detAct << ','
           << stat.detNoZ << ',' << stat.detAbrt << ',' << stat.totClus << std::endl;
    }

#else
    static void zeroStat() {}
    static void incTot(int) {}
    static void incReady() {}
    static void incSet() {}
    static void incAct() {}
    static void incNoZ() {}
    static void incAbrt() {}
    static void incClus(int) {}
    static void printStat() {}
#endif
  };
}  // namespace

class SiStripClusterizerFromZSRaw final : public edm::stream::EDProducer<> {
public:
  explicit SiStripClusterizerFromZSRaw(const edm::ParameterSet& conf)
      : onDemand(conf.getParameter<bool>("onDemand")),
        cabling_(nullptr),
        clusterizer_(StripClusterizerAlgorithmFactory::create(conf.getParameter<edm::ParameterSet>("Clusterizer"))),
        rawAlgos_(SiStripRawProcessingFactory::create(conf.getParameter<edm::ParameterSet>("Algorithms"))),
        doAPVEmulatorCheck_(conf.existsAs<bool>("DoAPVEmulatorCheck") ? conf.getParameter<bool>("DoAPVEmulatorCheck")
                                                                      : true),
        legacy_(conf.existsAs<bool>("LegacyUnpacker") ? conf.getParameter<bool>("LegacyUnpacker") : false),
        hybridZeroSuppressed_(conf.getParameter<bool>("HybridZeroSuppressed")) {
    productToken_ = consumes<FEDRawDataCollection>(conf.getParameter<edm::InputTag>("ProductLabel"));
    produces<edmNew::DetSetVector<SiStripCluster> >();
    assert(clusterizer_.get());
    assert(rawAlgos_.get());
    assert(!legacy_);
    assert(!onDemand);
  }

  void beginRun(const edm::Run&, const edm::EventSetup& es) override { initialize(es); }

  void produce(edm::Event& ev, const edm::EventSetup& es) override {
    initialize(es);

    // get raw data
    edm::Handle<FEDRawDataCollection> rawData;
    ev.getByToken(productToken_, rawData);

    auto output = std::make_unique<edmNew::DetSetVector<SiStripCluster>>();
    assert(!output->onDemand());

    output->reserve(15000, 24 * 10000);

    run(*rawData, *output);
    output->shrink_to_fit();
    COUT << output->dataSize() << " clusters from " << output->size() << " modules" << std::endl;
    ev.put(std::move(output));
  }

private:
  void initialize(const edm::EventSetup& es);

  void run(const FEDRawDataCollection& rawColl, edmNew::DetSetVector<SiStripCluster>& output);

private:
  bool onDemand;

  edm::EDGetTokenT<FEDRawDataCollection> productToken_;

  SiStripDetCabling const* cabling_;

  std::unique_ptr<StripClusterizerAlgorithm> clusterizer_;
  std::unique_ptr<SiStripRawProcessingAlgorithms> rawAlgos_;

  // March 2012: add flag for disabling APVe check in configuration
  bool doAPVEmulatorCheck_;

  bool legacy_;
  bool hybridZeroSuppressed_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripClusterizerFromZSRaw);


void SiStripClusterizerFromZSRaw::initialize(const edm::EventSetup& es) {
  (*clusterizer_).initialize(es);
  cabling_ = (*clusterizer_).cabling();
  (*rawAlgos_).initialize(es);
}

void SiStripClusterizerFromZSRaw::run(const FEDRawDataCollection& rawColl, edmNew::DetSetVector<SiStripCluster>& output) {
  ClusterFiller filler(rawColl, *clusterizer_, *rawAlgos_, doAPVEmulatorCheck_, legacy_, hybridZeroSuppressed_);

  // loop over good det in cabling
  for (auto const & idet : clusterizer_->allDets()) {
    StripClusterizerAlgorithm::output_t::FastFiller record(output, idet.detId);

    filler.fill(idet,record);

    if (record.empty())
      record.abort();

  }  // end loop over dets
}

namespace {
  template<typename OUT>
  void clustersFromZS(uint8_t const * data, int offset, int lenght, uint16_t stripOffset, 
                      StripClusterizerAlgorithm::Det const & det, OUT & out) {
    int ic=0;
    while (ic<lenght) {
       uint16_t firstStrip = stripOffset + data[(offset++) ^ 7];
       int clusSize = data[offset++ ^ 7];
       ic+=clusSize+2;
       int sum=0;
       int noise2=0;
       std::vector<uint8_t> adc(clusSize);
       for (int ic=0; ic<clusSize; ++ic) {
         uint16_t strip = firstStrip+ic;
         adc[ic]=data[(offset++) ^ 7];
         sum += adc[ic]; // no way it can overflow
         int noise = det.rawNoise(strip);
         noise2 += noise*noise;  // ditto
       }       
       if (4*sum*sum < noise2) continue;
       // calibrate and store;
       for (int ic=0; ic<clusSize; ++ic) {
         uint16_t strip = firstStrip+ic;
         adc[ic] = 0.5f+float(adc[ic])*det.weight(strip);
       }
       out.push_back(std::move(SiStripCluster(firstStrip,std::move(adc))));
    }
  }

}  // namespace

void ClusterFiller::fill(StripClusterizerAlgorithm::Det const & det, 
                         StripClusterizerAlgorithm::output_t::FastFiller& record) {
    incReady();

    auto idet = record.id();

    COUT << "filling " << idet << std::endl;

    if (!det.valid())
      return;

    incSet();

    // Loop over apv-pairs of det
    for (auto const conn : clusterizer.currentConnection(det)) {
      if
        UNLIKELY(!conn) continue;

      const uint16_t fedId = conn->fedId();

      // If fed id is null or connection is invalid continue
      if
        UNLIKELY(!fedId || !conn->isConnected()) { continue; }

      // If Fed hasnt already been initialised, extract data and initialise
      if (!buffers[fedId])
        buffers[fedId].reset(fillBuffer(fedId, rawColl).release());
      if (!buffers[fedId]) continue;
      auto buffer = buffers[fedId].get();

      buffer->setLegacyMode(legacy_);

      // check channel
      const uint8_t fedCh = conn->fedCh();

      if
        UNLIKELY(!buffer->channelGood(fedCh, doAPVEmulatorCheck)) {
          if (edm::isDebugEnabled()) {
            std::ostringstream ss;
            ss << "Problem unpacking channel " << fedCh << " on FED " << fedId;
            edm::LogWarning(sistrip::mlRawToCluster_) << ss.str();
          }
          continue;
        }

      // Determine APV std::pair number
      uint16_t ipair = conn->apvPairNumber();

      const sistrip::FEDReadoutMode mode = buffer->readoutMode();
      if (mode==sistrip::READOUT_MODE_ZERO_SUPPRESSED_LITE8 ||
           mode==sistrip::READOUT_MODE_ZERO_SUPPRESSED_LITE8_CMOVERRIDE) {
           auto const & ch = buffer->channel(fedCh);
           clustersFromZS(ch.data(), ch.offset() + 2, ch.length()-2, ipair * 256, det, record);
      } else {
       std::cout << "MODE NOT SUPPORTED" << std::endl;
      }
    }  // end loop over conn

    incAct();
    
    if (!record.empty())
      incNoZ();

//    COUT << "filled " << record.size() << std::endl;
//    for (auto const& cl : record)
//      COUT << cl.firstStrip() << ',' << cl.amplitudes().size() << std::endl;
    incClus(record.size());
}
