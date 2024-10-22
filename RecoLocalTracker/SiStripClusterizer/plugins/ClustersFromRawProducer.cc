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
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>
#include <memory>
#include <atomic>
#include <mutex>

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
    const auto st_buffer = sistrip::preconstructCheckFEDBuffer(rawData);
    if UNLIKELY (sistrip::FEDBufferStatusCode::SUCCESS != st_buffer) {
      if (edm::isDebugEnabled()) {
        edm::LogWarning(sistrip::mlRawToCluster_)
            << "[ClustersFromRawProducer::" << __func__ << "]" << st_buffer << " for FED ID " << fedId;
      }
      return buffer;
    }
    buffer = std::make_unique<sistrip::FEDBuffer>(rawData);
    const auto st_chan = buffer->findChannels();
    if UNLIKELY (sistrip::FEDBufferStatusCode::SUCCESS != st_chan) {
      if (edm::isDebugEnabled()) {
        edm::LogWarning(sistrip::mlRawToCluster_)
            << "Exception caught when creating FEDBuffer object for FED " << fedId << ": " << st_chan;
      }
      buffer.reset();
      return buffer;
    }
    if UNLIKELY (!buffer->doChecks(false)) {
      if (edm::isDebugEnabled()) {
        edm::LogWarning(sistrip::mlRawToCluster_)
            << "Exception caught when creating FEDBuffer object for FED " << fedId << ": FED Buffer check fails";
      }
      buffer.reset();
      return buffer;
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

  class ClusterFiller final : public StripClusterizerAlgorithm::output_t::Getter {
  public:
    ClusterFiller(const FEDRawDataCollection& irawColl,
                  StripClusterizerAlgorithm& iclusterizer,
                  SiStripRawProcessingAlgorithms& irawAlgos,
                  bool idoAPVEmulatorCheck,
                  bool legacy,
                  bool hybridZeroSuppressed)
        : rawColl(irawColl),
          clusterizer(iclusterizer),
          conditions(iclusterizer.conditions()),
          rawAlgos(irawAlgos),
          doAPVEmulatorCheck(idoAPVEmulatorCheck),
          legacy_(legacy),
          hybridZeroSuppressed_(hybridZeroSuppressed) {
      incTot(clusterizer.conditions().allDetIds().size());
      for (auto& d : done)
        d = nullptr;
    }

    ~ClusterFiller() override { printStat(); }

    void fill(StripClusterizerAlgorithm::output_t::TSFastFiller& record) const override;

  private:
    CMS_THREAD_GUARD(done) mutable std::unique_ptr<sistrip::FEDBuffer> buffers[1024];
    mutable std::atomic<sistrip::FEDBuffer*> done[1024];

    const FEDRawDataCollection& rawColl;

    StripClusterizerAlgorithm& clusterizer;
    const SiStripClusterizerConditions& conditions;
    SiStripRawProcessingAlgorithms& rawAlgos;

    // March 2012: add flag for disabling APVe check in configuration
    bool doAPVEmulatorCheck;

    bool legacy_;
    bool hybridZeroSuppressed_;

#ifdef VIDEBUG
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
      COUT << "VI clusters " << stat.totDet << ',' << stat.detReady << ',' << stat.detSet << ',' << stat.detAct << ','
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

class SiStripClusterizerFromRaw final : public edm::stream::EDProducer<> {
public:
  explicit SiStripClusterizerFromRaw(const edm::ParameterSet& conf)
      : onDemand(conf.getParameter<bool>("onDemand")),
        clusterizer_(StripClusterizerAlgorithmFactory::create(consumesCollector(),
                                                              conf.getParameter<edm::ParameterSet>("Clusterizer"))),
        rawAlgos_(SiStripRawProcessingFactory::create(conf.getParameter<edm::ParameterSet>("Algorithms"),
                                                      consumesCollector())),
        doAPVEmulatorCheck_(conf.getParameter<bool>("DoAPVEmulatorCheck")),
        legacy_(conf.getParameter<bool>("LegacyUnpacker")),
        hybridZeroSuppressed_(conf.getParameter<bool>("HybridZeroSuppressed")) {
    productToken_ = consumes<FEDRawDataCollection>(conf.getParameter<edm::InputTag>("ProductLabel"));
    produces<edmNew::DetSetVector<SiStripCluster> >();
    assert(clusterizer_.get());
    assert(rawAlgos_.get());
  }

  void produce(edm::Event& ev, const edm::EventSetup& es) override {
    initialize(es);

    // get raw data
    edm::Handle<FEDRawDataCollection> rawData;
    ev.getByToken(productToken_, rawData);

    std::unique_ptr<edmNew::DetSetVector<SiStripCluster> > output(
        onDemand ? new edmNew::DetSetVector<SiStripCluster>(
                       std::shared_ptr<edmNew::DetSetVector<SiStripCluster>::Getter>(std::make_shared<ClusterFiller>(
                           *rawData, *clusterizer_, *rawAlgos_, doAPVEmulatorCheck_, legacy_, hybridZeroSuppressed_)),
                       clusterizer_->conditions().allDetIds())
                 : new edmNew::DetSetVector<SiStripCluster>());

    if (onDemand)
      assert(output->onDemand());

    output->reserve(15000, 24 * 10000);

    if (!onDemand) {
      run(*rawData, *output);
      output->shrink_to_fit();
      COUT << output->dataSize() << " clusters from " << output->size() << " modules" << std::endl;
    }

    ev.put(std::move(output));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void initialize(const edm::EventSetup& es);

  void run(const FEDRawDataCollection& rawColl, edmNew::DetSetVector<SiStripCluster>& output);

private:
  bool onDemand;

  edm::EDGetTokenT<FEDRawDataCollection> productToken_;

  std::unique_ptr<StripClusterizerAlgorithm> clusterizer_;
  std::unique_ptr<SiStripRawProcessingAlgorithms> rawAlgos_;

  // March 2012: add flag for disabling APVe check in configuration
  bool doAPVEmulatorCheck_;

  bool legacy_;
  bool hybridZeroSuppressed_;
};

void SiStripClusterizerFromRaw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("ProductLabel", edm::InputTag("rawDataCollector"));
  desc.add<std::string>("ConditionsLabel", "");
  desc.add("onDemand", true);
  desc.add("DoAPVEmulatorCheck", true);
  desc.add("LegacyUnpacker", false);
  desc.add("HybridZeroSuppressed", false);

  edm::ParameterSetDescription clusterizer;
  StripClusterizerAlgorithmFactory::fillDescriptions(clusterizer);
  desc.add("Clusterizer", clusterizer);

  edm::ParameterSetDescription algorithms;
  SiStripRawProcessingFactory::fillDescriptions(algorithms);
  desc.add("Algorithms", algorithms);

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripClusterizerFromRaw);

void SiStripClusterizerFromRaw::initialize(const edm::EventSetup& es) {
  (*clusterizer_).initialize(es);
  (*rawAlgos_).initialize(es);
}

void SiStripClusterizerFromRaw::run(const FEDRawDataCollection& rawColl, edmNew::DetSetVector<SiStripCluster>& output) {
  ClusterFiller filler(rawColl, *clusterizer_, *rawAlgos_, doAPVEmulatorCheck_, legacy_, hybridZeroSuppressed_);

  // loop over good det in cabling
  for (auto idet : clusterizer_->conditions().allDetIds()) {
    StripClusterizerAlgorithm::output_t::TSFastFiller record(output, idet);

    filler.fill(record);

    if (record.empty())
      record.abort();
  }  // end loop over dets
}

namespace {
  class StripByStripAdder {
  public:
    typedef std::output_iterator_tag iterator_category;
    typedef void value_type;
    typedef void difference_type;
    typedef void pointer;
    typedef void reference;

    StripByStripAdder(StripClusterizerAlgorithm& clusterizer,
                      StripClusterizerAlgorithm::State& state,
                      StripClusterizerAlgorithm::output_t::TSFastFiller& record)
        : clusterizer_(clusterizer), state_(state), record_(record) {}

    StripByStripAdder& operator=(SiStripDigi digi) {
      clusterizer_.stripByStripAdd(state_, digi.strip(), digi.adc(), record_);
      return *this;
    }

    StripByStripAdder& operator*() { return *this; }
    StripByStripAdder& operator++() { return *this; }
    StripByStripAdder& operator++(int) { return *this; }

  private:
    StripClusterizerAlgorithm& clusterizer_;
    StripClusterizerAlgorithm::State& state_;
    StripClusterizerAlgorithm::output_t::TSFastFiller& record_;
  };

  template <typename Container>
  class ADC_back_inserter {
  public:
    ADC_back_inserter(Container& c) : c_(c) {}

    ADC_back_inserter& operator=(SiStripRawDigi digi) {
      c_.push_back(digi.adc());
      return *this;
    }
    ADC_back_inserter& operator*() { return *this; }
    ADC_back_inserter& operator++() { return *this; }
    ADC_back_inserter& operator++(int) { return *this; }

  private:
    Container& c_;
  };
}  // namespace

void ClusterFiller::fill(StripClusterizerAlgorithm::output_t::TSFastFiller& record) const {
  try {  // edmNew::CapacityExaustedException
    incReady();

    auto idet = record.id();

    COUT << "filling " << idet << std::endl;

    auto const& det = clusterizer.stripByStripBegin(idet);
    if (!det.valid())
      return;
    StripClusterizerAlgorithm::State state(det);

    incSet();
    record.reserve(16);
    // Loop over apv-pairs of det
    for (auto const conn : conditions.currentConnection(det)) {
      if UNLIKELY (!conn)
        continue;

      const uint16_t fedId = conn->fedId();

      // If fed id is null or connection is invalid continue
      if UNLIKELY (!fedId || !conn->isConnected()) {
        continue;
      }

      // If Fed hasnt already been initialised, extract data and initialise
      sistrip::FEDBuffer* buffer = done[fedId];
      if (!buffer) {
        buffer = fillBuffer(fedId, rawColl).release();
        if (!buffer) {
          continue;
        }
        sistrip::FEDBuffer* exp = nullptr;
        if (done[fedId].compare_exchange_strong(exp, buffer))
          buffers[fedId].reset(buffer);
        else {
          delete buffer;
          buffer = done[fedId];
        }
      }
      assert(buffer);

      buffer->setLegacyMode(legacy_);

      // check channel
      const uint8_t fedCh = conn->fedCh();

      if UNLIKELY (!buffer->channelGood(fedCh, doAPVEmulatorCheck)) {
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
      const sistrip::FEDLegacyReadoutMode lmode =
          legacy_ ? buffer->legacyReadoutMode() : sistrip::READOUT_MODE_LEGACY_INVALID;

      using namespace sistrip;
      if LIKELY (fedchannelunpacker::isZeroSuppressed(mode, legacy_, lmode)) {
        auto perStripAdder = StripByStripAdder(clusterizer, state, record);
        const auto isNonLite = fedchannelunpacker::isNonLiteZS(mode, legacy_, lmode);
        const uint8_t pCode = (isNonLite ? buffer->packetCode(legacy_, fedCh) : 0);
        auto st_ch = fedchannelunpacker::StatusCode::SUCCESS;
        if LIKELY (!hybridZeroSuppressed_) {
          st_ch = fedchannelunpacker::unpackZeroSuppressed(
              buffer->channel(fedCh), perStripAdder, ipair * 256, isNonLite, mode, legacy_, lmode, pCode);
        } else {
          const uint32_t id = conn->detId();
          edm::DetSet<SiStripDigi> unpDigis{id};
          unpDigis.reserve(256);
          st_ch = fedchannelunpacker::unpackZeroSuppressed(
              buffer->channel(fedCh), std::back_inserter(unpDigis), ipair * 256, isNonLite, mode, legacy_, lmode, pCode);
          if (fedchannelunpacker::StatusCode::SUCCESS == st_ch) {
            edm::DetSet<SiStripDigi> suppDigis{id};
            rawAlgos.suppressHybridData(unpDigis, suppDigis, ipair * 2);
            std::copy(std::begin(suppDigis), std::end(suppDigis), perStripAdder);
          }
        }
        if (fedchannelunpacker::StatusCode::SUCCESS != st_ch && edm::isDebugEnabled()) {
          edm::LogWarning(sistrip::mlRawToCluster_)
              << "Unordered clusters for channel " << fedCh << " on FED " << fedId << ": " << toString(st_ch);
          continue;
        }
      } else {
        auto st_ch = fedchannelunpacker::StatusCode::SUCCESS;
        if (fedchannelunpacker::isVirginRaw(mode, legacy_, lmode)) {
          std::vector<int16_t> digis;
          st_ch = fedchannelunpacker::unpackVirginRaw(
              buffer->channel(fedCh), ADC_back_inserter(digis), buffer->channel(fedCh).packetCode());
          if (fedchannelunpacker::StatusCode::SUCCESS == st_ch) {
            //process raw
            uint32_t id = conn->detId();
            edm::DetSet<SiStripDigi> zsdigis(id);
            //rawAlgos_->subtractorPed->subtract( id, ipair*256, digis);
            //rawAlgos_->subtractorCMN->subtract( id, digis);
            //rawAlgos_->suppressor->suppress( digis, zsdigis);
            uint16_t firstAPV = ipair * 2;
            rawAlgos.suppressVirginRawData(id, firstAPV, digis, zsdigis);
            for (const auto digi : zsdigis) {
              clusterizer.stripByStripAdd(state, digi.strip(), digi.adc(), record);
            }
          }
        } else if (fedchannelunpacker::isProcessedRaw(mode, legacy_, lmode)) {
          std::vector<int16_t> digis;
          st_ch = fedchannelunpacker::unpackProcessedRaw(buffer->channel(fedCh), ADC_back_inserter(digis));
          if (fedchannelunpacker::StatusCode::SUCCESS == st_ch) {
            //process raw
            uint32_t id = conn->detId();
            edm::DetSet<SiStripDigi> zsdigis(id);
            //rawAlgos_->subtractorCMN->subtract( id, digis);
            //rawAlgos_->suppressor->suppress( digis, zsdigis);
            uint16_t firstAPV = ipair * 2;
            rawAlgos.suppressProcessedRawData(id, firstAPV, digis, zsdigis);
            for (edm::DetSet<SiStripDigi>::const_iterator it = zsdigis.begin(); it != zsdigis.end(); it++) {
              clusterizer.stripByStripAdd(state, it->strip(), it->adc(), record);
            }
          }
        } else {
          edm::LogWarning(sistrip::mlRawToCluster_)
              << "[ClustersFromRawProducer::" << __func__ << "]"
              << " FEDRawData readout mode " << mode << " from FED id " << fedId << " not supported.";
        }
        if (fedchannelunpacker::StatusCode::SUCCESS != st_ch && edm::isDebugEnabled()) {
          edm::LogWarning(sistrip::mlRawToCluster_)
              << "[ClustersFromRawProducer::" << __func__ << "]" << toString(st_ch) << " from FED id " << fedId
              << " channel " << fedCh;
        }
      }
    }  // end loop over conn

    clusterizer.stripByStripEnd(state, record);

    incAct();

    if (record.full()) {
      edm::LogError(sistrip::mlRawToCluster_) << "too many Sistrip Clusters to fit space allocated for OnDemand for "
                                              << record.id() << ' ' << record.size();
      record.abort();
      incAbrt();
    }

    if (!record.empty())
      incNoZ();

    COUT << "filled " << record.size() << std::endl;
    for (auto const& cl : record)
      COUT << cl.firstStrip() << ',' << cl.amplitudes().size() << std::endl;
    incClus(record.size());
  } catch (edmNew::CapacityExaustedException const&) {
    edm::LogError(sistrip::mlRawToCluster_) << "too many Sistrip Clusters to fit space allocated for OnDemand";
  }
}
