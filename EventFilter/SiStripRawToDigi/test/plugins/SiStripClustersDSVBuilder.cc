#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripClustersDSVBuilder.h"

//FWCore
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Data Formats
#include "DataFormats/Common/interface/Handle.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripClustersDSVBuilder::SiStripClustersDSVBuilder( const edm::ParameterSet& conf ) :

  inputModuleLabel_(conf.getUntrackedParameter<string>("InputModuleLabel","")),
  outputProductLabel_(conf.getUntrackedParameter<string>("OutputProductLabel",""))
  
{
  LogTrace(mlRawToCluster_)
    << "[SiStripClustersDSVBuilder::" 
    << __func__ 
    << "]"
    << " Constructing object...";
  
  produces< DSV >(outputProductLabel_);
}

// -----------------------------------------------------------------------------
/** */
SiStripClustersDSVBuilder::~SiStripClustersDSVBuilder() {

  LogTrace(mlRawToCluster_)
    << "[SiStripClustersDSVBuilder::" 
    << __func__ 
    << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
void SiStripClustersDSVBuilder::beginJob( const edm::EventSetup& setup) {

  LogTrace(mlRawToCluster_) 
    << "[SiStripClustersDSVBuilder::"
    << __func__ 
    << "]";
}

// -----------------------------------------------------------------------------
void SiStripClustersDSVBuilder::endJob() {;}

// -----------------------------------------------------------------------------
/** */
void SiStripClustersDSVBuilder::produce( edm::Event& event, 
					 const edm::EventSetup& setup ) {  
 
  //Retrieve RefGetter with demand from event
  edm::Handle< RefGetter > demandclusters;
  event.getByLabel(inputModuleLabel_,demandclusters);
 
  //Construct product
  auto_ptr<DSV> dsv(new DSV);
  RefGetter::const_iterator iregion = demandclusters->begin();
  for(;iregion!=demandclusters->end();++iregion) {
    vector<SiStripCluster>::const_iterator icluster = iregion->first;
    for (;icluster!=iregion->second;icluster++) {
      DetSet& detset = dsv->find_or_insert(icluster->geographicalId());
      detset.push_back(*icluster);
    }
  }
  //add to event
  event.put(dsv);
}
