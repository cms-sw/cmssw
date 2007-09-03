#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClustersRoI.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
#include "CLHEP/Random/RandFlat.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripRawToClustersRoI::SiStripRawToClustersRoI( const edm::ParameterSet& conf ) :

  inputModuleLabel_(conf.getUntrackedParameter<string>("InputModuleLabel","")),
  cabling_(),
  nlayers_(conf.getUntrackedParameter<int>("Layers",-1)),
  random_(conf.getUntrackedParameter<bool>("Random",false)),
  all_(conf.getUntrackedParameter<bool>("All",true)),
  electron_(conf.getUntrackedParameter<bool>("Electron",false)),
  deta_(conf.getUntrackedParameter<double>("DeltaEta",0.2)),
  dphi_(conf.getUntrackedParameter<double>("DeltaPhi",0.2))
  
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
  edm::Handle< LazyGetter > lazygetter;
  event.getByLabel(inputModuleLabel_,"",lazygetter);
  
  // Construct default RefGetter object
  std::auto_ptr<RefGetter> refgetter(new RefGetter());

  // Fill RefGetter with regions of interest
  if (random_) 
    {
      random(*refgetter,lazygetter);
    }

  if (all_) 
    {
      all(*refgetter,lazygetter);
    }

  if (electron_) 
    {
      edm::Handle<reco::SuperClusterCollection> barrelsclustersisolated;
      edm::Handle<reco::SuperClusterCollection> endcapsclustersisolated;
      event.getByLabel("hltIslandSuperClustersL1Isolated","islandEndcapSuperClusters",endcapsclustersisolated);
      event.getByLabel("hltHybridSuperClustersL1Isolated","",barrelsclustersisolated);
      superclusters(*barrelsclustersisolated,*refgetter,lazygetter);
      superclusters(*endcapsclustersisolated,*refgetter,lazygetter);
    }
  
  // Add to event
  event.put(refgetter);
}

bool SiStripRawToClustersRoI::physicalLayer(SiStripRegionCabling::SubDet& subdet, uint32_t& layer) const {
  int signedlayer = static_cast<int>(SiStripRegionCabling::physicalLayer(subdet,layer));
  return (nlayers_ == -1 || signedlayer < nlayers_) ? true : false;
}

void SiStripRawToClustersRoI::random(RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {
  
  uint32_t total = cabling_->getRegionCabling().size();
  uint32_t required = (uint32_t)(RandFlat::shoot()*(total+1));
  for (uint32_t iregion = 0; iregion < required; iregion++) {
    for (uint32_t isubdet = 0; isubdet < SiStripRegionCabling::ALLSUBDETS; isubdet++) {
      SiStripRegionCabling::SubDet subdet = static_cast<SiStripRegionCabling::SubDet>(isubdet);
      for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
	if (!physicalLayer(subdet,ilayer)) break;
	cabling_->updateSiStripRefGetter<SiStripCluster>(refgetter,lazygetter,SiStripRegionCabling::elementIndex(iregion,subdet,ilayer));
      }
    }
  }
}

void SiStripRawToClustersRoI::all(RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {
  
  for (uint32_t iregion = 0; iregion < cabling_->getRegionCabling().size(); iregion++) {
    for (uint32_t isubdet = 0; isubdet < SiStripRegionCabling::ALLSUBDETS; isubdet++) {
      SiStripRegionCabling::SubDet subdet = static_cast<SiStripRegionCabling::SubDet>(isubdet);
      for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
	if (!physicalLayer(subdet,ilayer)) break;
	cabling_->updateSiStripRefGetter<SiStripCluster>(refgetter,lazygetter,SiStripRegionCabling::elementIndex(iregion,subdet,ilayer));
      }
    }
  }
}

void SiStripRawToClustersRoI::superclusters(const reco::SuperClusterCollection& coll, RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {
  
  for (reco::SuperClusterCollection::const_iterator iclust = coll.begin(); iclust!=coll.end(); iclust++) {
    SiStripRegionCabling::Position position(iclust->seed()->position().eta(),iclust->seed()->position().phi());
    for (uint32_t isubdet = 0; isubdet < SiStripRegionCabling::ALLSUBDETS; isubdet++) {
      SiStripRegionCabling::SubDet subdet = static_cast<SiStripRegionCabling::SubDet>(isubdet);
      for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
	if (!physicalLayer(subdet,ilayer)) break;
	cabling_->updateSiStripRefGetter<SiStripCluster>(refgetter,lazygetter,position,deta_,dphi_,subdet,ilayer);
      }
    }
  }
}

