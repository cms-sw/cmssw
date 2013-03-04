#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClusters.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithmFactory.h"
#include "RecoLocalTracker/SiStripZeroSuppression/interface/SiStripRawProcessingFactory.h"

using namespace std;

namespace sistrip {

RawToClusters::RawToClusters( const edm::ParameterSet& conf ) :
  productLabel_(conf.getParameter<edm::InputTag>("ProductLabel")),
  cabling_(0),
  cacheId_(0),
  clusterizer_(StripClusterizerAlgorithmFactory::create(conf.getParameter<edm::ParameterSet>("Clusterizer"))),
  rawAlgos_(SiStripRawProcessingFactory::create(conf.getParameter<edm::ParameterSet>("Algorithms"))),
  doAPVEmulatorCheck_(conf.existsAs<bool>("DoAPVEmulatorCheck") ? conf.getParameter<bool>("DoAPVEmulatorCheck") : true)
{
  if ( edm::isDebugEnabled() ) {
    LogTrace("SiStripRawToCluster")
      << "[RawToClusters::" << __func__ << "]"
      << " Constructing object...";
  }
  produces<LazyGetter>();

}

  RawToClusters::~RawToClusters() {
    if ( edm::isDebugEnabled() ) {
      LogTrace("SiStripRawToCluster")
	<< "[RawToClusters::" << __func__ << "]"
	<< " Destructing object...";
    }
  }

  void RawToClusters::beginRun( const edm::Run&, const edm::EventSetup& setup) {
    updateCabling( setup );  
    clusterizer_->initialize(setup);
    rawAlgos_->initialize(setup);
  }

  void RawToClusters::produce( edm::Event& event,const edm::EventSetup& setup ) {
  
    // update cabling
    updateCabling( setup );  
    clusterizer_->initialize( setup );
    rawAlgos_->initialize( setup );
  
    // get raw data
    edm::Handle<FEDRawDataCollection> buffers;
    event.getByLabel( productLabel_, buffers ); 

    // create lazy unpacker
    boost::shared_ptr<LazyUnpacker> unpacker( new LazyUnpacker( *cabling_, *clusterizer_, *rawAlgos_, *buffers ) );

    // propagate the parameter doAPVEmulatorCheck_ to the unpacker.
    unpacker->doAPVEmulatorCheck(doAPVEmulatorCheck_);

    // create lazy getter
    std::auto_ptr<LazyGetter> collection( new LazyGetter( cabling_->getRegionCabling().size() * SiStripRegionCabling::ALLSUBDETS * SiStripRegionCabling::ALLLAYERS, unpacker ) );
  
    // add collection to the event
    event.put( collection );
  
  }

  void RawToClusters::updateCabling( const edm::EventSetup& setup ) {

    uint32_t cache_id = setup.get<SiStripRegionCablingRcd>().cacheIdentifier();
    if ( cacheId_ != cache_id ) {
      edm::ESHandle<SiStripRegionCabling> c;
      setup.get<SiStripRegionCablingRcd>().get( c );
      cabling_ = c.product();
      cacheId_ = cache_id;
    }
  }

}
