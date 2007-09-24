#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClustersRoI.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
#include "CLHEP/Random/RandFlat.h"

using namespace std;
using namespace sistrip;

SiStripRawToClustersRoI::SiStripRawToClustersRoI( const edm::ParameterSet& conf ) :

  cabling_(),
  inputModuleLabel_(conf.getUntrackedParameter<string>("InputModuleLabel","")),
  nlayers_(conf.getUntrackedParameter<int>("Layers",-1)),
  global_(conf.getUntrackedParameter<bool>("Global",true)),
  random_(conf.getUntrackedParameter<bool>("Random",false)),
  electrons_(conf.getUntrackedParameter<bool>("Electrons",false)),
  muons_(conf.getUntrackedParameter<bool>("Muons",false)),
  taus_(conf.getUntrackedParameter<bool>("Taus",false)),
  bjets_(conf.getUntrackedParameter<bool>("Bjets",false)),
  electronBarrelModule_(conf.getUntrackedParameter<string>("ElectronBarrelModule","")),
  electronBarrelProduct_(conf.getUntrackedParameter<string>("ElectronBarrelProduct","")),
  electronEndcapModule_(conf.getUntrackedParameter<string>("ElectronEndcapModule","")),
  electronEndcapProduct_(conf.getUntrackedParameter<string>("ElectronEndcapProduct","")),
  muonModule_(conf.getUntrackedParameter<string>("MuonModule","")),
  muonProduct_(conf.getUntrackedParameter<string>("MuonProduct","")),
  tauModule_(conf.getUntrackedParameter<string>("TauModule","")),
  tauProduct_(conf.getUntrackedParameter<string>("TauProduct","")),
  bjetModule_(conf.getUntrackedParameter<string>("BjetModule","")),
  bjetProduct_(conf.getUntrackedParameter<string>("BjetProduct","")),
  electrondeta_(conf.getUntrackedParameter<double>("ElectronEtaWindow",0.2)),
  electrondphi_(conf.getUntrackedParameter<double>("ElectronPhiWindow",0.2)),
  muondeta_(conf.getUntrackedParameter<double>("MuonEtaWindow",0.2)),
  muondphi_(conf.getUntrackedParameter<double>("MuonPhiWindow",0.2)),
  taudeta_(conf.getUntrackedParameter<double>("TauEtaWindow",0.2)),
  taudphi_(conf.getUntrackedParameter<double>("TauPhiWindow",0.2)),
  bjetdeta_(conf.getUntrackedParameter<double>("BjetEtaWindow",0.2)),
  bjetdphi_(conf.getUntrackedParameter<double>("BjetPhiWindow",0.2))
{
  LogTrace(mlRawToCluster_)
    << "[SiStripRawToClustersRoI::" 
    << __func__ 
    << "]"
    << " Constructing object...";
  
  produces< RefGetter >();
}

SiStripRawToClustersRoI::~SiStripRawToClustersRoI() {

  LogTrace(mlRawToCluster_)
    << "[SiStripRawToClustersRoI::" 
    << __func__ 
    << "]"
    << " Destructing object...";
}

void SiStripRawToClustersRoI::beginJob( const edm::EventSetup& setup) {

  LogTrace(mlRawToCluster_) 
    << "[SiStripRawToClustersRoI::"
    << __func__ 
    << "]";

  setup.get<SiStripRegionCablingRcd>().get(cabling_);
}

void SiStripRawToClustersRoI::endJob() {}

void SiStripRawToClustersRoI::produce( edm::Event& event, const edm::EventSetup& setup ) {
  
  // Retrieve unpacking tool from event
  edm::Handle< LazyGetter > lazygetter;
  event.getByLabel(inputModuleLabel_,"",lazygetter);
  
  // Construct default RefGetter object
  std::auto_ptr<RefGetter> refgetter(new RefGetter());
  
  // Fill RefGetter with regions of interest
  if (global_) global(*refgetter,lazygetter);

  if (random_) random(*refgetter,lazygetter);
  
  if (electrons_) {
    try {
    edm::Handle<reco::SuperClusterCollection> barrelcollection;
    edm::Handle<reco::SuperClusterCollection> endcapcollection;
    event.getByLabel(electronBarrelModule_,electronBarrelProduct_,barrelcollection);
    event.getByLabel(electronEndcapModule_,electronEndcapProduct_,endcapcollection);
    superclusters(*barrelcollection,*refgetter,lazygetter);
    superclusters(*endcapcollection,*refgetter,lazygetter);
    } catch(...) {}
  }

  if (muons_) {
    try {
    edm::Handle<reco::TrackCollection> collection;
    event.getByLabel(muonModule_,muonProduct_,collection);
    muons(*collection,*refgetter,lazygetter);
    } catch(...) {}
  }
  
  if (taus_) {
    try {
    edm::Handle<reco::CaloJetCollection> collection;
    event.getByLabel(tauModule_,tauProduct_,collection);
    taus(*collection,*refgetter,lazygetter);
    } catch(...) {}
  }
  
  if (bjets_) {
    try {
    edm::Handle<reco::CaloJetCollection> collection;
    event.getByLabel(bjetModule_,bjetProduct_,collection);
    bjets(*collection,*refgetter,lazygetter);
    } catch(...) {}
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
  uint32_t required = static_cast<uint32_t>(RandFlat::shoot()*(total+1));
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

void SiStripRawToClustersRoI::global(RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {
  
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

void SiStripRawToClustersRoI::superclusters(const reco::SuperClusterCollection& collection, RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {
  
  for (reco::SuperClusterCollection::const_iterator icollection = collection.begin(); icollection!=collection.end(); icollection++) {
    SiStripRegionCabling::Position position(icollection->eta(),icollection->phi());
    for (uint32_t isubdet = 0; isubdet < SiStripRegionCabling::ALLSUBDETS; isubdet++) {
      SiStripRegionCabling::SubDet subdet = static_cast<SiStripRegionCabling::SubDet>(isubdet);
      for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
	if (!physicalLayer(subdet,ilayer)) break;
	cabling_->updateSiStripRefGetter<SiStripCluster>(refgetter,lazygetter,position,electrondeta_,electrondphi_,subdet,ilayer);
      }
    }    
  }
}

void SiStripRawToClustersRoI::muons(const reco::TrackCollection& collection, RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {

  for (reco::TrackCollection::const_iterator icollection = collection.begin(); icollection!=collection.end(); icollection++) {
    SiStripRegionCabling::Position position(icollection->outerEta(),icollection->outerPhi());
    for (uint32_t isubdet = 0; isubdet < SiStripRegionCabling::ALLSUBDETS; isubdet++) {
      SiStripRegionCabling::SubDet subdet = static_cast<SiStripRegionCabling::SubDet>(isubdet);
      for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
	if (!physicalLayer(subdet,ilayer)) break;
	cabling_->updateSiStripRefGetter<SiStripCluster>(refgetter,lazygetter,position,muondeta_,muondphi_,subdet,ilayer);
      }
    }    
  }
}

void SiStripRawToClustersRoI::taus(const reco::CaloJetCollection& collection, RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {

  for (reco::CaloJetCollection::const_iterator icollection = collection.begin(); icollection!=collection.end(); icollection++) {
    SiStripRegionCabling::Position position(icollection->eta(),icollection->phi());
    for (uint32_t isubdet = 0; isubdet < SiStripRegionCabling::ALLSUBDETS; isubdet++) {
      SiStripRegionCabling::SubDet subdet = static_cast<SiStripRegionCabling::SubDet>(isubdet);
      for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
	if (!physicalLayer(subdet,ilayer)) break;
	cabling_->updateSiStripRefGetter<SiStripCluster>(refgetter,lazygetter,position,taudeta_,taudphi_,subdet,ilayer);
      }
    }    
  }
}

void SiStripRawToClustersRoI::bjets(const reco::CaloJetCollection& collection, RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {

  for (reco::CaloJetCollection::const_iterator icollection = collection.begin(); icollection!=collection.end(); icollection++) {
    SiStripRegionCabling::Position position(icollection->eta(),icollection->phi());
    for (uint32_t isubdet = 0; isubdet < SiStripRegionCabling::ALLSUBDETS; isubdet++) {
      SiStripRegionCabling::SubDet subdet = static_cast<SiStripRegionCabling::SubDet>(isubdet);
      for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
	if (!physicalLayer(subdet,ilayer)) break;
	cabling_->updateSiStripRefGetter<SiStripCluster>(refgetter,lazygetter,position,bjetdeta_,bjetdphi_,subdet,ilayer);
      }
    }    
  }
}

