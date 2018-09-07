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
#include<iostream>
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
    if UNLIKELY( !rawData.data() ) {
      if (edm::isDebugEnabled()) {
        edm::LogWarning(sistrip::mlRawToCluster_)
          << "[ClustersFromRawProducer::" 
          << __func__ 
          << "]"
          << " NULL pointer to FEDRawData for FED id " 
          << fedId;
      }
      return buffer;
    }	
    
    // Check on FEDRawData size
    if UNLIKELY( !rawData.size() ) {
      if (edm::isDebugEnabled()) {
        edm::LogWarning(sistrip::mlRawToCluster_)
	  << "[ClustersFromRawProducer::" 
	  << __func__ << "]"
	  << " FEDRawData has zero size for FED id " 
	  << fedId;
      }
      return buffer;
    }
    
    // construct FEDBuffer
    try {
      buffer.reset(new sistrip::FEDBuffer(rawData.data(),rawData.size()));
      if UNLIKELY(!buffer->doChecks(false)) throw cms::Exception("FEDBuffer") << "FED Buffer check fails for FED ID" << fedId << ".";
    }
    catch (const cms::Exception& e) { 
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
  
  
  class ClusterFiller final : public StripClusterizerAlgorithm::output_t::Getter {
  public:
    ClusterFiller(const FEDRawDataCollection& irawColl, 
                  StripClusterizerAlgorithm & iclusterizer, 
                  SiStripRawProcessingAlgorithms & irawAlgos, 
                  bool idoAPVEmulatorCheck, 
                  bool hybridZeroSuppressed):
      rawColl(irawColl),
      clusterizer(iclusterizer),
      rawAlgos(irawAlgos),
      doAPVEmulatorCheck(idoAPVEmulatorCheck),
      hybridZeroSuppressed_(hybridZeroSuppressed){
        incTot(clusterizer.allDetIds().size());
        for (auto & d : done) d=nullptr;
      }
    
    
    ~ClusterFiller() override { printStat();}
    
    void fill(StripClusterizerAlgorithm::output_t::TSFastFiller & record) override;
    
  private:
    
    
    std::unique_ptr<sistrip::FEDBuffer> buffers[1024];
    std::atomic<sistrip::FEDBuffer*> done[1024];
    
    
    const FEDRawDataCollection& rawColl;
    
    StripClusterizerAlgorithm & clusterizer;
    SiStripRawProcessingAlgorithms & rawAlgos;
    
    
    // March 2012: add flag for disabling APVe check in configuration
    bool doAPVEmulatorCheck; 

    bool hybridZeroSuppressed_;
    
    
#ifdef VIDEBUG
    struct Stat {
      Stat() : totDet(0), detReady(0),detSet(0),detAct(0),detNoZ(0),detAbrt(0),totClus(0){}
      std::atomic<int> totDet; // all dets
      std::atomic<int> detReady; // dets "updated"
      std::atomic<int> detSet;  // det actually set not empty
      std::atomic<int> detAct;  // det actually set with content
      std::atomic<int> detNoZ;  // det actually set with content
      std::atomic<int> detAbrt;  // det aborted
      std::atomic<int> totClus; // total number of clusters
    };
    
    mutable Stat stat;
    // void zeroStat() const { stat = std::move(Stat()); }
    void incTot(int n) const { stat.totDet=n;}
    void incReady() const { stat.detReady++;}
    void incSet() const { stat.detSet++;}
    void incAct() const { stat.detAct++;}
    void incNoZ() const { stat.detNoZ++;}
    void incAbrt() const { stat.detAbrt++;}
    void incClus(int n) const { stat.totClus+=n;}
    void printStat() const {
      COUT << "VI clusters " << stat.totDet <<','<< stat.detReady <<','<< stat.detSet <<','<< stat.detAct<<','<< stat.detNoZ <<','<<stat.detAbrt <<','<<stat.totClus << std::endl;
    }
    
#else
    static void zeroStat(){}
    static void incTot(int){}
    static void incReady() {}
    static void incSet() {}
    static void incAct() {}
    static void incNoZ() {}
    static void incAbrt(){}
    static void incClus(int){}
    static void printStat(){}
#endif
    
  };
  
  
} // namespace



class SiStripClusterizerFromRaw final : public edm::stream::EDProducer<>  {
  
 public:
  
  explicit SiStripClusterizerFromRaw(const edm::ParameterSet& conf) :
    onDemand(conf.getParameter<bool>("onDemand")),
    cabling_(nullptr),
    clusterizer_(StripClusterizerAlgorithmFactory::create(conf.getParameter<edm::ParameterSet>("Clusterizer"))),
    rawAlgos_(SiStripRawProcessingFactory::create(conf.getParameter<edm::ParameterSet>("Algorithms"))),
    doAPVEmulatorCheck_(conf.existsAs<bool>("DoAPVEmulatorCheck") ? conf.getParameter<bool>("DoAPVEmulatorCheck") : true),
    hybridZeroSuppressed_(conf.getParameter<bool>("HybridZeroSuppressed"))
      {
	productToken_ = consumes<FEDRawDataCollection>(conf.getParameter<edm::InputTag>("ProductLabel"));
	produces< edmNew::DetSetVector<SiStripCluster> > ();
	assert(clusterizer_.get());
	assert(rawAlgos_.get());
      }
  

  void beginRun( const edm::Run&, const edm::EventSetup& es) override {
    initialize(es);
  }
  
  
  void produce(edm::Event& ev, const edm::EventSetup& es) override {
    
    initialize(es);
    
    // get raw data
    edm::Handle<FEDRawDataCollection> rawData;
    ev.getByToken( productToken_, rawData); 
    
    
    std::unique_ptr< edmNew::DetSetVector<SiStripCluster> > output(
      onDemand ?
        new edmNew::DetSetVector<SiStripCluster>(
          std::shared_ptr<edmNew::DetSetVector<SiStripCluster>::Getter>(
              std::make_shared<ClusterFiller>(*rawData, *clusterizer_, *rawAlgos_,
                                              doAPVEmulatorCheck_, hybridZeroSuppressed_)),
          clusterizer_->allDetIds())
      : new edmNew::DetSetVector<SiStripCluster>()
      );
    
    if(onDemand) assert(output->onDemand());

    output->reserve(15000,24*10000);


    if (!onDemand) {
      run(*rawData, *output);
      output->shrink_to_fit();   
      COUT << output->dataSize() << " clusters from " 
	   << output->size()     << " modules" 
	   << std::endl;
    }
   
    ev.put(std::move(output));

  }

private:

  void initialize(const edm::EventSetup& es);

  void run(const FEDRawDataCollection& rawColl, edmNew::DetSetVector<SiStripCluster> & output);


 private:

  bool  onDemand;

  edm::EDGetTokenT<FEDRawDataCollection> productToken_;  
  
  SiStripDetCabling const * cabling_;
  
  std::unique_ptr<StripClusterizerAlgorithm> clusterizer_;
  std::unique_ptr<SiStripRawProcessingAlgorithms> rawAlgos_;
  
  
  // March 2012: add flag for disabling APVe check in configuration
  bool doAPVEmulatorCheck_; 

  bool hybridZeroSuppressed_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripClusterizerFromRaw);




void SiStripClusterizerFromRaw::initialize(const edm::EventSetup& es) {

  (*clusterizer_).initialize(es);
  cabling_ = (*clusterizer_).cabling();
  (*rawAlgos_).initialize(es);

}

void SiStripClusterizerFromRaw::run(const FEDRawDataCollection& rawColl,
				     edmNew::DetSetVector<SiStripCluster> & output) {
  
  ClusterFiller filler(rawColl, *clusterizer_, *rawAlgos_, doAPVEmulatorCheck_, hybridZeroSuppressed_);
  
  // loop over good det in cabling
  for ( auto idet : clusterizer_->allDetIds()) {

    StripClusterizerAlgorithm::output_t::TSFastFiller record(output, idet);	
    
    filler.fill(record);
    
    if(record.empty()) record.abort();

  } // end loop over dets
}

namespace {
  template<typename OUT>
  OUT unpackZS(const sistrip::FEDChannel& chan, sistrip::FEDReadoutMode mode, uint16_t stripOffset, OUT out)
  {
    using namespace sistrip;
    switch ( mode ) {
      case READOUT_MODE_ZERO_SUPPRESSED_LITE8:
      case READOUT_MODE_ZERO_SUPPRESSED_LITE8_CMOVERRIDE:
      { auto unpacker = FEDZSChannelUnpacker::zeroSuppressedLiteModeUnpacker(chan);
        while (unpacker.hasData()) { *out++ = SiStripDigi(stripOffset+unpacker.sampleNumber(), unpacker.adc()); unpacker++; }
      } break;
      case READOUT_MODE_ZERO_SUPPRESSED_LITE10:
      case READOUT_MODE_ZERO_SUPPRESSED_LITE10_CMOVERRIDE:
      { auto unpacker = FEDBSChannelUnpacker::zeroSuppressedLiteModeUnpacker(chan, 10);
        while (unpacker.hasData()) { *out++ = SiStripDigi(stripOffset+unpacker.sampleNumber(), unpacker.adc()); unpacker++; }
      } break;
      case READOUT_MODE_ZERO_SUPPRESSED:
      case READOUT_MODE_ZERO_SUPPRESSED_FAKE:
      {
        switch ( chan.packetCode() ) {
          case PACKET_CODE_ZERO_SUPPRESSED:
          { auto unpacker = FEDZSChannelUnpacker::zeroSuppressedModeUnpacker(chan);
            while (unpacker.hasData()) { *out++ = SiStripDigi(stripOffset+unpacker.sampleNumber(), unpacker.adc()); unpacker++; }
          } break;
          case PACKET_CODE_ZERO_SUPPRESSED10:
          { auto unpacker = FEDBSChannelUnpacker::zeroSuppressedModeUnpacker(chan, 10);
            while (unpacker.hasData()) { *out++ = SiStripDigi(stripOffset+unpacker.sampleNumber(), unpacker.adc()); unpacker++; }
          } break;
          case PACKET_CODE_ZERO_SUPPRESSED8_BOTBOT:
          { auto unpacker = FEDBSChannelUnpacker::zeroSuppressedModeUnpacker(chan, 8);
            while (unpacker.hasData()) { *out++ = SiStripDigi(stripOffset+unpacker.sampleNumber(), unpacker.adc()<<2); unpacker++; }
          } break;
          case PACKET_CODE_ZERO_SUPPRESSED8_TOPBOT:
          { auto unpacker = FEDBSChannelUnpacker::zeroSuppressedModeUnpacker(chan, 8);
            while (unpacker.hasData()) { *out++ = SiStripDigi(stripOffset+unpacker.sampleNumber(), unpacker.adc()<<1); unpacker++; }
          } break;
          default:
            edm::LogWarning(mlRawToCluster_) << "[ClustersFromRawProducer::" << __func__ << "]"
              << " invalid packet code " << chan.packetCode() << " for zero-suppressed.";
        }
      } break;
      case READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT:
      case READOUT_MODE_ZERO_SUPPRESSED_LITE8_TOPBOT_CMOVERRIDE:
      { auto unpacker = FEDZSChannelUnpacker::zeroSuppressedLiteModeUnpacker(chan);
        while (unpacker.hasData()) { *out++ = SiStripDigi(stripOffset+unpacker.sampleNumber(), unpacker.adc()<<1); unpacker++; }
      } break;
      case READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT:
      case READOUT_MODE_ZERO_SUPPRESSED_LITE8_BOTBOT_CMOVERRIDE:
      { auto unpacker = FEDZSChannelUnpacker::zeroSuppressedLiteModeUnpacker(chan);
        while (unpacker.hasData()) { *out++ = SiStripDigi(stripOffset+unpacker.sampleNumber(), unpacker.adc()<<2); unpacker++; }
      } break;
      default:;
    }
    return out;
  }

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

    StripByStripAdder& operator= ( SiStripDigi digi )
    {
      clusterizer_.stripByStripAdd(state_, digi.strip(), digi.adc(), record_);
      return *this;
    }

    StripByStripAdder& operator*  ()    { return *this; }
    StripByStripAdder& operator++ ()    { return *this; }
    StripByStripAdder& operator++ (int) { return *this; }
  private:
    StripClusterizerAlgorithm& clusterizer_;
    StripClusterizerAlgorithm::State& state_;
    StripClusterizerAlgorithm::output_t::TSFastFiller& record_;
  };
}

void ClusterFiller::fill(StripClusterizerAlgorithm::output_t::TSFastFiller & record) {
try { // edmNew::CapacityExaustedException
  incReady();

  auto idet= record.id();

  COUT << "filling " << idet << std::endl;

  auto const & det = clusterizer.stripByStripBegin(idet);
  if (!det.valid()) return; 
  StripClusterizerAlgorithm::State state(det);

  incSet();

  // Loop over apv-pairs of det
  for (auto const conn : clusterizer.currentConnection(det)) {
    if UNLIKELY(!conn) continue;
    
    const uint16_t fedId = conn->fedId();
    
    // If fed id is null or connection is invalid continue
    if UNLIKELY( !fedId || !conn->isConnected() ) { continue; }    
    

    // If Fed hasnt already been initialised, extract data and initialise
    sistrip::FEDBuffer * buffer = done[fedId];
    if (!buffer) { 
      buffer = fillBuffer(fedId, rawColl).release();
      if (!buffer) { continue;}
      sistrip::FEDBuffer * exp = nullptr;
      if (done[fedId].compare_exchange_strong(exp, buffer)) buffers[fedId].reset(buffer);
      else { delete buffer; buffer = done[fedId]; }
    }
    assert(buffer);

    // check channel
    const uint8_t fedCh = conn->fedCh();
    
    if UNLIKELY(!buffer->channelGood(fedCh,doAPVEmulatorCheck)) {
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

    if LIKELY( ( mode > sistrip::READOUT_MODE_VIRGIN_RAW ) && ( mode < sistrip::READOUT_MODE_SPY ) && ( mode != sistrip::READOUT_MODE_PROC_RAW ) ) {
      // ZS modes
      try {
        auto perStripAdder = StripByStripAdder(clusterizer, state, record);
        if LIKELY( ! hybridZeroSuppressed_ ) {
          unpackZS(buffer->channel(fedCh), mode, ipair*256, perStripAdder);
        } else {
          const uint32_t id = conn->detId();
          edm::DetSet<SiStripDigi> unpDigis{id}; unpDigis.reserve(256);
          unpackZS(buffer->channel(fedCh), mode, ipair*256, std::back_inserter(unpDigis));
          SiStripRawProcessingAlgorithms::digivector_t workRawDigis;
          rawAlgos.convertHybridDigiToRawDigiVector(unpDigis, workRawDigis);
          edm::DetSet<SiStripDigi> suppDigis{id};
          rawAlgos.suppressHybridData(id, ipair*2, workRawDigis, suppDigis);
          std::copy(std::begin(suppDigis), std::end(suppDigis), perStripAdder);
        }
      } catch (edmNew::CapacityExaustedException) {
        throw;
      } catch (const cms::Exception& e) {
        if (edm::isDebugEnabled()) {
          edm::LogWarning(sistrip::mlRawToCluster_) << "Unordered clusters for channel " << fedCh << " on FED " << fedId << ": " << e.what();
        }
        continue;
      }

    } else if ( mode == sistrip::READOUT_MODE_VIRGIN_RAW ) {

      std::vector<int16_t> digis;
      switch ( buffer->channel(fedCh).packetCode() ) {
        case sistrip::PACKET_CODE_VIRGIN_RAW:
        { auto unpacker = sistrip::FEDRawChannelUnpacker::virginRawModeUnpacker(buffer->channel(fedCh));
          while (unpacker.hasData()) { digis.push_back(unpacker.adc()); unpacker++; }
        } break;
        case sistrip::PACKET_CODE_VIRGIN_RAW10:
        { auto unpacker = sistrip::FEDBSChannelUnpacker::virginRawModeUnpacker(buffer->channel(fedCh), 10);
          while (unpacker.hasData()) { digis.push_back(unpacker.adc()); unpacker++; }
        } break;
        case sistrip::PACKET_CODE_VIRGIN_RAW8_BOTBOT:
        { auto unpacker = sistrip::FEDBSChannelUnpacker::virginRawModeUnpacker(buffer->channel(fedCh), 8);
          while (unpacker.hasData()) { digis.push_back(unpacker.adc()<<2); unpacker++; }
        } break;
        case sistrip::PACKET_CODE_VIRGIN_RAW8_TOPBOT:
        { auto unpacker = sistrip::FEDBSChannelUnpacker::virginRawModeUnpacker(buffer->channel(fedCh), 8);
          while (unpacker.hasData()) { digis.push_back(unpacker.adc()<<1); unpacker++; }
        } break;
        default:
          edm::LogWarning(sistrip::mlRawToCluster_) << "[ClustersFromRawProducer::" << __func__ << "]"
            << " invalid packet code " << buffer->channel(fedCh).packetCode() << " for virgin raw.";
      }
      //process raw
      uint32_t id = conn->detId();
      edm::DetSet<SiStripDigi> zsdigis(id);
      //rawAlgos_->subtractorPed->subtract( id, ipair*256, digis);
      //rawAlgos_->subtractorCMN->subtract( id, digis);
      //rawAlgos_->suppressor->suppress( digis, zsdigis);
      uint16_t firstAPV = ipair*2;
      rawAlgos.suppressVirginRawData(id, firstAPV,digis, zsdigis);
      for ( const auto digi : zsdigis ) {
        clusterizer.stripByStripAdd(state, digi.strip(), digi.adc(), record);
      }

    } else if ( mode == sistrip::READOUT_MODE_PROC_RAW ) {

      // create unpacker
      sistrip::FEDRawChannelUnpacker unpacker = sistrip::FEDRawChannelUnpacker::procRawModeUnpacker(buffer->channel(fedCh));

      // unpack
      std::vector<int16_t> digis;
      while (unpacker.hasData()) {
        digis.push_back(unpacker.adc());
        unpacker++;
      }

      //process raw
      uint32_t id = conn->detId();
      edm::DetSet<SiStripDigi> zsdigis(id);
      //rawAlgos_->subtractorCMN->subtract( id, digis);
      //rawAlgos_->suppressor->suppress( digis, zsdigis);
      uint16_t firstAPV = ipair*2;
      rawAlgos.suppressProcessedRawData(id, firstAPV,digis, zsdigis);
      for( edm::DetSet<SiStripDigi>::const_iterator it = zsdigis.begin(); it!=zsdigis.end(); it++) {
        clusterizer.stripByStripAdd(state, it->strip(), it->adc(), record);
      }
    } else {
      edm::LogWarning(sistrip::mlRawToCluster_)
        << "[ClustersFromRawProducer::" << __func__ << "]"
        << " FEDRawData readout mode " << mode << " from FED id " << fedId << " not supported.";
      continue;
    }
  } // end loop over conn

  clusterizer.stripByStripEnd(state,record);
  
  incAct();
 
  if (record.full()) {
    edm::LogError(sistrip::mlRawToCluster_) << "too many Sistrip Clusters to fit space allocated for OnDemand for " << record.id() << ' ' << record.size();
    record.abort();
    incAbrt();
  }
  
  if(!record.empty()) incNoZ();

  COUT << "filled " << record.size() << std::endl;
  for ( auto const & cl : record ) COUT << cl.firstStrip() << ','<<  cl.amplitudes().size() << std::endl;
  incClus(record.size());
} catch (edmNew::CapacityExaustedException const&) {
  edm::LogError(sistrip::mlRawToCluster_) << "too many Sistrip Clusters to fit space allocated for OnDemand";
}  

}


