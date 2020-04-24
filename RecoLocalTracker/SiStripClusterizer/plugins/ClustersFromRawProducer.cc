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
    if unlikely( !rawData.data() ) {
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
    if unlikely( !rawData.size() ) {
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
      if unlikely(!buffer->doChecks(false)) throw cms::Exception("FEDBuffer") << "FED Buffer check fails for FED ID" << fedId << ".";
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
		  bool idoAPVEmulatorCheck):
      rawColl(irawColl),
      clusterizer(iclusterizer),
      rawAlgos(irawAlgos),
      doAPVEmulatorCheck(idoAPVEmulatorCheck){
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
    doAPVEmulatorCheck_(conf.existsAs<bool>("DoAPVEmulatorCheck") ? conf.getParameter<bool>("DoAPVEmulatorCheck") : true)
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
    
    
    std::unique_ptr< edmNew::DetSetVector<SiStripCluster> > 
      output( onDemand ?
	      new edmNew::DetSetVector<SiStripCluster>(std::shared_ptr<edmNew::DetSetVector<SiStripCluster>::Getter>(std::make_shared<ClusterFiller>(*rawData, *clusterizer_, 
																	 *rawAlgos_, doAPVEmulatorCheck_)
														       ), 
						       clusterizer_->allDetIds())
	      : new edmNew::DetSetVector<SiStripCluster>());
    
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
  
  ClusterFiller filler(rawColl, *clusterizer_, *rawAlgos_, doAPVEmulatorCheck_);
  
  // loop over good det in cabling
  for ( auto idet : clusterizer_->allDetIds()) {

    StripClusterizerAlgorithm::output_t::TSFastFiller record(output, idet);	
    
    filler.fill(record);
    
    if(record.empty()) record.abort();

  } // end loop over dets
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
    if unlikely(!conn) continue;
    
    const uint16_t fedId = conn->fedId();
    
    // If fed id is null or connection is invalid continue
    if unlikely( !fedId || !conn->isConnected() ) { continue; }    
    

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
    
    if unlikely(!buffer->channelGood(fedCh,doAPVEmulatorCheck)) {
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
    
    
    if likely(mode == sistrip::READOUT_MODE_ZERO_SUPPRESSED_LITE10 || mode == sistrip::READOUT_MODE_ZERO_SUPPRESSED_LITE8) { 
	
	try {
	  // create unpacker
	  sistrip::FEDZSChannelUnpacker unpacker = sistrip::FEDZSChannelUnpacker::zeroSuppressedLiteModeUnpacker(buffer->channel(fedCh));
	  
	  // unpack
	  clusterizer.addFed(state,unpacker,ipair,record);
	  /*
            while (unpacker.hasData()) {
	    clusterizer.stripByStripAdd(unpacker.sampleNumber()+ipair*256,unpacker.adc(),record);
	    unpacker++;
            }
	  */
	} catch (edmNew::CapacityExaustedException) {
          throw;
        } catch (const cms::Exception& e) {
	  if (edm::isDebugEnabled()) {
	    std::ostringstream ss;
	    ss << "Unordered clusters for channel " << fedCh << " on FED " << fedId << ": " << e.what();
	    edm::LogWarning(sistrip::mlRawToCluster_) << ss.str();
	  }                                               
	  continue;
	}
    } else {
      
      if (mode == sistrip::READOUT_MODE_ZERO_SUPPRESSED || mode == sistrip::READOUT_MODE_ZERO_SUPPRESSED_FAKE ) { 
	try {
	  // create unpacker
	  sistrip::FEDZSChannelUnpacker unpacker = sistrip::FEDZSChannelUnpacker::zeroSuppressedModeUnpacker(buffer->channel(fedCh));
	  
	  // unpack
	  clusterizer.addFed(state,unpacker,ipair,record);
	  /*
	    while (unpacker.hasData()) {
	    clusterizer.stripByStripAdd(unpacker.sampleNumber()+ipair*256,unpacker.adc(),record);
	    unpacker++;
	    }
	  */
	} catch (edmNew::CapacityExaustedException) {
           throw;
        }catch (const cms::Exception& e) {
	  if (edm::isDebugEnabled()) {
	    std::ostringstream ss;
	    ss << "Unordered clusters for channel " << fedCh << " on FED " << fedId << ": " << e.what();
	    edm::LogWarning(sistrip::mlRawToCluster_) << ss.str();
	  }
	  continue;
	}
      } else if (mode == sistrip::READOUT_MODE_VIRGIN_RAW ) {
	
	// create unpacker
	sistrip::FEDRawChannelUnpacker unpacker = sistrip::FEDRawChannelUnpacker::virginRawModeUnpacker(buffer->channel(fedCh));
	
	// unpack
	std::vector<int16_t> digis;
	while (unpacker.hasData()) {
	  digis.push_back(unpacker.adc());
	  unpacker++;
	}
	
	//process raw
	uint32_t id = conn->detId();
	edm::DetSet<SiStripDigi> zsdigis(id);
	//rawAlgos_->subtractorPed->subtract( id, ipair*256, digis);
	//rawAlgos_->subtractorCMN->subtract( id, digis);
	//rawAlgos_->suppressor->suppress( digis, zsdigis);
	uint16_t firstAPV = ipair*2;
	rawAlgos.SuppressVirginRawData(id, firstAPV,digis, zsdigis);
 	for( edm::DetSet<SiStripDigi>::const_iterator it = zsdigis.begin(); it!=zsdigis.end(); it++) {
	  clusterizer.stripByStripAdd(state, it->strip(), it->adc(), record);
	}
      }
      
      else if (mode == sistrip::READOUT_MODE_PROC_RAW ) {
	
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
	rawAlgos.SuppressProcessedRawData(id, firstAPV,digis, zsdigis);
	for( edm::DetSet<SiStripDigi>::const_iterator it = zsdigis.begin(); it!=zsdigis.end(); it++) {
	  clusterizer.stripByStripAdd(state, it->strip(), it->adc(), record);
	}
      } else {
	edm::LogWarning(sistrip::mlRawToCluster_)
	  << "[ClustersFromRawProducer::" 
	  << __func__ << "]"
	  << " FEDRawData readout mode "
	  << mode
	  << " from FED id "
	  << fedId 
	  << " not supported."; 
	continue;
      }
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
} catch (edmNew::CapacityExaustedException) {
  edm::LogError(sistrip::mlRawToCluster_) << "too many Sistrip Clusters to fit space allocated for OnDemand";
}  

}


