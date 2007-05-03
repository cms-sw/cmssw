#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripRawToClustersDummyUnpacker.h"

//FWCore
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Data Formats
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCommon/test/stubs/SiStripRefGetter.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripRawToClustersDummyUnpacker::SiStripRawToClustersDummyUnpacker( const edm::ParameterSet& conf ) :

  inputModuleLabel_(conf.getUntrackedParameter<string>("InputModuleLabel",""))
  
{
  LogTrace(mlRawToCluster_)
    << "[SiStripRawToClustersDummyUnpacker::" 
    << __func__ 
    << "]"
    << " Constructing object...";
  
  produces< RefGetter >();
}

// -----------------------------------------------------------------------------
/** */
SiStripRawToClustersDummyUnpacker::~SiStripRawToClustersDummyUnpacker() {

  LogTrace(mlRawToCluster_)
    << "[SiStripRawToClustersDummyUnpacker::" 
    << __func__ 
    << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
void SiStripRawToClustersDummyUnpacker::beginJob( const edm::EventSetup& setup) {

  LogTrace(mlRawToCluster_) 
    << "[SiStripRawToClustersDummyUnpacker::"
    << __func__ 
    << "]";
}

// -----------------------------------------------------------------------------
void SiStripRawToClustersDummyUnpacker::endJob() {;}

// -----------------------------------------------------------------------------
/** */
void SiStripRawToClustersDummyUnpacker::produce( edm::Event& event, 
						 const edm::EventSetup& setup ) {  
  //Retrieve RefGetter with demand from event
  edm::Handle< RefGetter > demandclusters;
  event.getByLabel(inputModuleLabel_,demandclusters);
  
  RefGetter::const_iterator iregion = demandclusters->begin();
  for(;iregion!=demandclusters->end();++iregion) {
    vector<SiStripCluster>::const_iterator icluster = iregion->first;
    for (;icluster!=iregion->second;icluster++) {
      icluster->geographicalId();
    }
  }
}
