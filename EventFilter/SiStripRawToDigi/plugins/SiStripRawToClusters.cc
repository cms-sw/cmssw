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

using namespace std;

OldSiStripRawToClusters::OldSiStripRawToClusters( const edm::ParameterSet& conf ) :
  productLabel_(conf.getParameter<edm::InputTag>("ProductLabel")),
  cabling_(0),
  cacheId_(0),
  clusterizer_(0)
{
  if ( edm::isDebugEnabled() ) {
    LogTrace("SiStripRawToCluster")
      << "[OldSiStripRawToClusters::" << __func__ << "]"
      << " Constructing object...";
  }
  clusterizer_ = new SiStripClusterizerFactory(conf);
  produces<LazyGetter>();
}

OldSiStripRawToClusters::~OldSiStripRawToClusters() {
  if (clusterizer_) { delete clusterizer_; }
  if ( edm::isDebugEnabled() ) {
    LogTrace("SiStripRawToCluster")
      << "[OldSiStripRawToClusters::" << __func__ << "]"
      << " Destructing object...";
  }
}

void OldSiStripRawToClusters::beginJob( const edm::EventSetup& setup) {
  //@@ unstable behaviour if uncommented!
  //updateCabling( setup );  
  //clusterizer_->eventSetup(setup);
}

void OldSiStripRawToClusters::beginRun( edm::Run&, const edm::EventSetup& setup) {
  updateCabling( setup );  
  clusterizer_->eventSetup(setup);
}

void OldSiStripRawToClusters::produce( edm::Event& event,const edm::EventSetup& setup ) {
  
  // update cabling
  updateCabling( setup );  
  clusterizer_->eventSetup( setup );
  
  // get raw data
  edm::Handle<FEDRawDataCollection> buffers;
  event.getByLabel( productLabel_, buffers ); 

  // create lazy unpacker
  boost::shared_ptr<LazyUnpacker> unpacker( new LazyUnpacker( *cabling_, *clusterizer_, *buffers ) );

  // create lazy getter
  std::auto_ptr<LazyGetter> collection( new LazyGetter( cabling_->getRegionCabling().size() * SiStripRegionCabling::ALLSUBDETS * SiStripRegionCabling::ALLLAYERS, unpacker ) );
  
  // add collection to the event
  event.put( collection );
  
}

void OldSiStripRawToClusters::updateCabling( const edm::EventSetup& setup ) {

  uint32_t cache_id = setup.get<SiStripRegionCablingRcd>().cacheIdentifier();
  if ( cacheId_ != cache_id ) {
    edm::ESHandle<SiStripRegionCabling> c;
    setup.get<SiStripRegionCablingRcd>().get( c );
    cabling_ = c.product();
    cacheId_ = cache_id;
  }
}

namespace sistrip {

RawToClusters::RawToClusters( const edm::ParameterSet& conf ) :
  productLabel_(conf.getParameter<edm::InputTag>("ProductLabel")),
  cabling_(0),
  cacheId_(0),
  clusterizer_(StripClusterizerAlgorithmFactory::create(conf.getParameter<edm::ParameterSet>("Clusterizer")))
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

  void RawToClusters::beginJob( const edm::EventSetup& setup) {
    //@@ unstable behaviour if uncommented!
    //updateCabling( setup );  
    //clusterizer_->initialize(setup);
  }

  void RawToClusters::beginRun( edm::Run&, const edm::EventSetup& setup) {
    updateCabling( setup );  
    clusterizer_->initialize(setup);
  }

  void RawToClusters::produce( edm::Event& event,const edm::EventSetup& setup ) {
  
    // update cabling
    updateCabling( setup );  
    clusterizer_->initialize( setup );
  
    // get raw data
    edm::Handle<FEDRawDataCollection> buffers;
    event.getByLabel( productLabel_, buffers ); 

    // create lazy unpacker
    boost::shared_ptr<LazyUnpacker> unpacker( new LazyUnpacker( *cabling_, *clusterizer_, *buffers ) );

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
