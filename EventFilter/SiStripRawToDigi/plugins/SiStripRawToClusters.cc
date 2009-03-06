#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClusters.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace std;
using namespace sistrip;

SiStripRawToClusters::SiStripRawToClusters( const edm::ParameterSet& conf ) :

  productLabel_(conf.getUntrackedParameter<string>("ProductLabel","source")),
  productInstance_(conf.getUntrackedParameter<string>("ProductInstance","")),
  cabling_(0),
  cacheId_(0),
  clusterizer_(0)
{
  if ( edm::isDebugEnabled() ) {
    LogTrace(mlRawToDigi_)
      << "[SiStripRawToClusters::" << __func__ << "]"
      << " Constructing object...";
  }
  clusterizer_ = new SiStripClusterizerFactory(conf);
  produces<LazyGetter>();
}

SiStripRawToClusters::~SiStripRawToClusters() {
  if (clusterizer_) { delete clusterizer_; }
  if ( edm::isDebugEnabled() ) {
    LogTrace(mlRawToDigi_)
      << "[SiStripRawToClusters::" << __func__ << "]"
      << " Destructing object...";
  }
}

void SiStripRawToClusters::beginJob( const edm::EventSetup& setup) {
  //@@ unstable behaviour if uncommented!
  //updateCabling( setup );  
  //clusterizer_->eventSetup(setup);
}

void SiStripRawToClusters::beginRun( edm::Run&, const edm::EventSetup& setup) {
  updateCabling( setup );  
  clusterizer_->eventSetup(setup);
}

void SiStripRawToClusters::produce( edm::Event& event,const edm::EventSetup& setup ) {
  
  updateCabling( setup );  
  clusterizer_->eventSetup( setup );
  
  edm::Handle<FEDRawDataCollection> buffers;
  event.getByLabel( productLabel_, productInstance_, buffers ); 
  boost::shared_ptr<LazyUnpacker> unpacker( new LazyUnpacker( *cabling_, *clusterizer_, *buffers ) );
  std::auto_ptr<LazyGetter> collection( new LazyGetter( cabling_->getRegionCabling().size() * 
							SiStripRegionCabling::ALLSUBDETS * 
							SiStripRegionCabling::ALLLAYERS,
							unpacker ) );
  
  event.put( collection );
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripRawToClusters::updateCabling( const edm::EventSetup& setup ) {
  uint32_t cache_id = setup.get<SiStripRegionCablingRcd>().cacheIdentifier();
  if ( cacheId_ != cache_id ) {
    edm::ESHandle<SiStripRegionCabling> c;
    setup.get<SiStripRegionCablingRcd>().get( c );
    cabling_ = c.product();
    cacheId_ = cache_id;
  }
}
