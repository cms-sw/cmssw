#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClustersRoI.h"

//FWCore
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DataFormats
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

//CalibTracker
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"

//CLHEP
#include "CLHEP/Random/RandFlat.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripRawToClustersRoI::SiStripRawToClustersRoI( const edm::ParameterSet& conf ) :

  inputModuleLabel_(conf.getUntrackedParameter<string>("InputModuleLabel","")),
  cabling_(),
  random_(conf.getUntrackedParameter<bool>("Random",false)),
  all_(conf.getUntrackedParameter<bool>("All",true)),
  electron_(conf.getUntrackedParameter<bool>("Electron",false)),
  dR_(conf.getUntrackedParameter<double>("DeltaR",0.5))
  
{
  LogTrace(mlRawToCluster_)
    << "[SiStripRawToClustersRoI::" 
    << __func__ 
    << "]"
    << " Constructing object...";
  
  produces< RefGetter >();
}

// -----------------------------------------------------------------------------
/** */
SiStripRawToClustersRoI::~SiStripRawToClustersRoI() {

  LogTrace(mlRawToCluster_)
    << "[SiStripRawToClustersRoI::" 
    << __func__ 
    << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
void SiStripRawToClustersRoI::beginJob( const edm::EventSetup& setup) {

  LogTrace(mlRawToCluster_) 
    << "[SiStripRawToClustersRoI::"
    << __func__ 
    << "]";

 //Fill cabling
  setup.get<SiStripRegionCablingRcd>().get(cabling_);
}

// -----------------------------------------------------------------------------
void SiStripRawToClustersRoI::endJob() {;}

// -----------------------------------------------------------------------------
/** */
void SiStripRawToClustersRoI::produce( edm::Event& event, 
					    const edm::EventSetup& setup ) {
  
  // Retrieve unpacking tool from event
  edm::Handle<edm::SiStripLazyGetter<SiStripCluster> > getter;
  event.getByLabel(inputModuleLabel_,"",getter);
  
  // Create regions of interest vector
  SiStripRegionCabling::Regions demand;
  demand.reserve(cabling_->getRegionCabling().size());

  if (random_) random(demand);
  if (all_) all(demand);
  
  if (electron_) {
  edm::Handle<reco::SuperClusterCollection> barrelsclusters;
  edm::Handle<reco::SuperClusterCollection> endcapsclusters;
  event.getByLabel("correctedHybridSuperClusters","",barrelsclusters);
  event.getByLabel("correctedIslandEndcapSuperClusters","",endcapsclusters);
  superclusters(*barrelsclusters,demand);
  superclusters(*endcapsclusters,demand);
  }

  // Add regions of interest to RefGetter object
  std::auto_ptr<RefGetter> regions(new RefGetter(getter,demand));

  // Add to event
  event.put(regions);
}


void SiStripRawToClustersRoI::random(SiStripRegionCabling::Regions& regions) const {
  
  uint32_t total = cabling_->getRegionCabling().size();
  uint32_t required = (uint32_t)(RandFlat::shoot()*(total+1));

  for (uint32_t iregion = 0; 
       iregion < required; 
       iregion++) {
    
    for (uint32_t isubdet = 0; 
         isubdet < cabling_->getRegionCabling()[iregion].size(); 
         isubdet++) {
      
      for (uint32_t ilayer = 0; 
	   ilayer < cabling_->getRegionCabling()[iregion][isubdet].size(); 
	   ilayer++) {
	
	uint32_t index = SiStripRegionCabling::elementIndex(iregion,static_cast<SiStripRegionCabling::SubDet>(isubdet),ilayer);
	regions.push_back(index);
      }
    }
  }
}

void SiStripRawToClustersRoI::all(SiStripRegionCabling::Regions& regions) const {
  
  uint32_t total = cabling_->getRegionCabling().size();
  
  for (uint32_t iregion = 0; 
       iregion < total; 
       iregion++) {
    
    for (uint32_t isubdet = 0; 
         isubdet < cabling_->getRegionCabling()[iregion].size(); 
         isubdet++) {
      
      for (uint32_t ilayer = 0; 
	   ilayer < cabling_->getRegionCabling()[iregion][isubdet].size(); 
	   ilayer++) {

	uint32_t index = SiStripRegionCabling::elementIndex(iregion,static_cast<SiStripRegionCabling::SubDet>(isubdet),ilayer);
	regions.push_back(index);
      }
    }
  }
}

void SiStripRawToClustersRoI::superclusters(const reco::SuperClusterCollection& coll, SiStripRegionCabling::Regions& regions) const {
  
  reco::SuperClusterCollection::const_iterator iclust = coll.begin();
  for (;
       iclust!=coll.end();
       iclust++) {
    
    SiStripRegionCabling::Position position(iclust->seed()->position().eta(),iclust->seed()->position().phi());
    SiStripRegionCabling::Regions newregions = cabling_->regions(position,dR_);
    
    SiStripRegionCabling::Regions::const_iterator iregion = newregions.begin();
    for (;
	 iregion!=newregions.end();
	 iregion++) {
      
      for (uint32_t isubdet = 0; 
	   isubdet < cabling_->getRegionCabling()[*iregion].size(); 
	   isubdet++) {
	
	for (uint32_t ilayer = 0; 
	     ilayer < cabling_->getRegionCabling()[*iregion][isubdet].size(); 
	     ilayer++) {
	  
	  uint32_t index = SiStripRegionCabling::elementIndex(*iregion,static_cast<SiStripRegionCabling::SubDet>(isubdet),ilayer);
	  regions.push_back(index);
	}
      }
    }
  }
}
  
