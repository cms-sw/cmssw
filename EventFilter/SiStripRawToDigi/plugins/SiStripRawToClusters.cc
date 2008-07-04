#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClusters.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

using namespace std;
using namespace sistrip;

SiStripRawToClusters::SiStripRawToClusters( const edm::ParameterSet& conf ) :

  productLabel_(conf.getUntrackedParameter<string>("ProductLabel","source")),
  productInstance_(conf.getUntrackedParameter<string>("ProductInstance","")),
  cabling_(),
  clusterizer_(0)
  
{
  if ( edm::isDebugEnabled() ) {
    LogTrace(mlRawToDigi_)
      << "[SiStripRawToClusters::" 
      << __func__ 
      << "]"
    << " Constructing object...";
  }

  clusterizer_ = new SiStripClusterizerFactory(conf);
  produces< LazyGetter >();
}

SiStripRawToClusters::~SiStripRawToClusters() {

  if (clusterizer_) delete clusterizer_;

  if ( edm::isDebugEnabled() ) {
    LogTrace(mlRawToDigi_)
      << "[SiStripRawToClusters::" 
      << __func__ 
      << "]"
      << " Destructing object...";
  }
}

void SiStripRawToClusters::beginJob( const edm::EventSetup& setup) {

  clusterizer_->eventSetup(setup);
}

void SiStripRawToClusters::endJob() {}

void SiStripRawToClusters::produce( edm::Event& event,const edm::EventSetup& setup ) {
  
  edm::Handle<FEDRawDataCollection> buffers;
  event.getByLabel( productLabel_, productInstance_, buffers ); 
  setup.get<SiStripRegionCablingRcd>().get(cabling_);
  boost::shared_ptr<LazyUnpacker> unpacker(new LazyUnpacker(*cabling_,*clusterizer_,*buffers));
  std::auto_ptr<LazyGetter> collection(new LazyGetter(cabling_->getRegionCabling().size()*SiStripRegionCabling::ALLSUBDETS*SiStripRegionCabling::ALLLAYERS,unpacker));
  event.put(collection);
}

