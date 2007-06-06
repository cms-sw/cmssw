#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClusters.h"

//FWCore
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Data Formats
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripRawToClusters::SiStripRawToClusters( const edm::ParameterSet& conf ) :

  allregions_(),
  productLabel_(conf.getUntrackedParameter<string>("ProductLabel","source")),
  productInstance_(conf.getUntrackedParameter<string>("ProductInstance","")),
  cabling_(),
  clusterizer_(0),
  dumpFrequency_(conf.getUntrackedParameter<int>("FedBufferDumpFreq",0)),
  triggerFedId_(conf.getUntrackedParameter<int>("TriggerFedId",0))
  
{
  LogTrace(mlRawToDigi_)
    << "[SiStripRawToClusters::" 
    << __func__ 
    << "]"
    << " Constructing object...";
  
  clusterizer_ = new SiStripClusterizerFactory(conf);

  produces< edm::SiStripLazyGetter<SiStripCluster> >();
  produces< RefGetter >();
}

// -----------------------------------------------------------------------------
/** */
SiStripRawToClusters::~SiStripRawToClusters() {

  if (clusterizer_) delete clusterizer_;

  LogTrace(mlRawToDigi_)
    << "[SiStripRawToClusters::" 
    << __func__ 
    << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
void SiStripRawToClusters::beginJob( const edm::EventSetup& setup) {

  //Fill cabling
  setup.get<SiStripRegionCablingRcd>().get(cabling_);

  //Configure clusterizer factory and pass to LazyGetter
  clusterizer_->eventSetup(setup);

  //Fill allregions_ record
  uint32_t nregions = cabling_->getRegionCabling().size();
  allregions_.reserve(nregions);
  for (uint32_t iregion = 0;iregion < nregions;iregion++) {
    allregions_.push_back(iregion);
  }
}

// -----------------------------------------------------------------------------
void SiStripRawToClusters::endJob() {;}

// -----------------------------------------------------------------------------
/** 
*/
void SiStripRawToClusters::produce( edm::Event& event, 
				    const edm::EventSetup& setup ) {


  //Retrieve FED raw data (by label, which is "source" by default)
  edm::Handle<FEDRawDataCollection> buffers;
  event.getByLabel( productLabel_, productInstance_, buffers ); 
  
  //Construct.
  boost::shared_ptr<SiStripRawToClustersLazyUnpacker> getter(new SiStripRawToClustersLazyUnpacker(*cabling_,*clusterizer_,*buffers));

  //Store SiStripLazyGetter in event.
  std::auto_ptr< edm::SiStripLazyGetter<SiStripCluster> > collection(new edm::SiStripLazyGetter<SiStripCluster>(getter));
  edm::OrphanHandle< edm::SiStripLazyGetter<SiStripCluster> > pcollection = event.put(collection);

  //Store SiStripRefGetter for global unpacking in event. 
  std::auto_ptr<RefGetter> rcollection(new RefGetter(pcollection,allregions_));
  event.put(rcollection);

}

