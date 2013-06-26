#include "EventFilter/SiStripRawToDigi/plugins/SiStripRawToClustersRoI.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
#include "CLHEP/Random/RandFlat.h"

using namespace std;
using namespace sistrip;

SiStripRawToClustersRoI::SiStripRawToClustersRoI(const edm::ParameterSet& conf) :
  cabling_(0),
  cacheId_(0),
  allregions_(),
  nlayers_(conf.getUntrackedParameter<int>("Layers",-1)),
  global_(conf.getUntrackedParameter<bool>("Global",true)),
  random_(conf.getUntrackedParameter<bool>("Random",false)),
  electrons_(conf.getUntrackedParameter<bool>("Electrons",false)),
  muons_(conf.getUntrackedParameter<bool>("Muons",false)),
  taujets_(conf.getUntrackedParameter<bool>("TauJets",false)),
  bjets_(conf.getUntrackedParameter<bool>("BJets",false)),
  siStripLazyGetter_(conf.getParameter<edm::InputTag>("SiStripLazyGetter")),
  electronBarrelL2_(conf.getParameter<edm::InputTag>("ElectronBarrelL2")),
  electronEndcapL2_(conf.getParameter<edm::InputTag>("ElectronEndcapL2")),
  muonL2_(conf.getParameter<edm::InputTag>("MuonL2")),
  taujetL2_(conf.getParameter<edm::InputTag>("TauJetL2")),
  bjetL2_(conf.getParameter<edm::InputTag>("BJetL2")),
  electrondeta_(conf.getUntrackedParameter<double>("ElectronEtaWindow",0.2)),
  electrondphi_(conf.getUntrackedParameter<double>("ElectronPhiWindow",0.2)),
  muondeta_(conf.getUntrackedParameter<double>("MuonEtaWindow",0.2)),
  muondphi_(conf.getUntrackedParameter<double>("MuonPhiWindow",0.2)),
  taujetdeta_(conf.getUntrackedParameter<double>("TauJetEtaWindow",0.2)),
  taujetdphi_(conf.getUntrackedParameter<double>("TauJetPhiWindow",0.2)),
  bjetdeta_(conf.getUntrackedParameter<double>("BJetEtaWindow",0.2)),
  bjetdphi_(conf.getUntrackedParameter<double>("BJetPhiWindow",0.2))
{
  produces< RefGetter >();
}

SiStripRawToClustersRoI::~SiStripRawToClustersRoI() {}

void SiStripRawToClustersRoI::beginRun( const edm::Run&, const edm::EventSetup& setup) {
  updateCabling( setup );  
}

void SiStripRawToClustersRoI::produce(edm::Event& event, const edm::EventSetup& setup) {
  
  updateCabling( setup );  
  
  edm::Handle< LazyGetter > lazygetter;
  event.getByLabel(siStripLazyGetter_,lazygetter);
  
  /// All regions 
  
  if (global_) {
    std::auto_ptr<RefGetter> globalrefgetter(new RefGetter(lazygetter,allregions_));
    /*//unpack
      for (RefGetter::const_iterator iReg = globalrefgetter->begin(); iReg != globalrefgetter->end(); iReg++) *iReg;*/
    event.put(globalrefgetter);
    return;
  }
  
  std::auto_ptr<RefGetter> refgetter(new RefGetter(allregions_.size()));
  refgetter->reserve(10000);
  
  /// Random region number. Starts from 0.
  
  if (random_) {random(*refgetter,lazygetter);}
  
  /// Seeded by L2 electrons.

  if (electrons_) {
	edm::Handle<reco::SuperClusterCollection> barrelcollection;
	edm::Handle<reco::SuperClusterCollection> endcapcollection;
	event.getByLabel(electronBarrelL2_,barrelcollection);
	event.getByLabel(electronEndcapL2_,endcapcollection);
	if (barrelcollection.isValid()) {
	  electrons(*barrelcollection,*refgetter,lazygetter);
	}
	if (endcapcollection.isValid()) {
	  electrons(*endcapcollection,*refgetter,lazygetter);
	}
  }

  /// Seeded by L2 muons.

  if (muons_) {
	edm::Handle<reco::TrackCollection> collection;
	event.getByLabel(muonL2_,collection);
	if (collection.isValid()) {
	  muons(*collection,*refgetter,lazygetter);
	}
    }
  
  /// Seeded by L2 taujets.

  if (taujets_) {
	edm::Handle<reco::CaloJetCollection> collection;
	event.getByLabel(taujetL2_,collection);
	if (collection.isValid()) {
	  taujets(*collection,*refgetter,lazygetter);
	}
    }
  
  /// Seeded by L2 bjets.

  if (bjets_) {
	edm::Handle<reco::CaloJetCollection> collection;
	event.getByLabel(bjetL2_,collection);
	if (collection.isValid()) {
	  bjets(*collection,*refgetter,lazygetter);
	}
    }
  
  event.put(refgetter);
}

// -----------------------------------------------------------------------------
/** */
void SiStripRawToClustersRoI::updateCabling( const edm::EventSetup& setup ) {
  uint32_t cache_id = setup.get<SiStripRegionCablingRcd>().cacheIdentifier();
  if ( cacheId_ != cache_id ) {
    edm::ESHandle<SiStripRegionCabling> c;
    setup.get<SiStripRegionCablingRcd>().get( c );
    cabling_ = c.product();
    cacheId_ = cache_id;
    allregions_.clear();
    allregions_.reserve( cabling_->getRegionCabling().size() );
    for (uint32_t iregion=0;iregion<cabling_->getRegionCabling().size();iregion++) {
      for (uint32_t isubdet=0;isubdet<cabling_->getRegionCabling()[iregion].size();isubdet++) {  
	for (uint32_t ilayer=0;ilayer<cabling_->getRegionCabling()[iregion][isubdet].size();ilayer++) {
	  uint32_t index = SiStripRegionCabling::elementIndex(iregion,static_cast<SubDet>(isubdet),ilayer);
	  allregions_.push_back(index);
	}
      }
    }
  }
}

bool SiStripRawToClustersRoI::physicalLayer(SubDet& subdet, uint32_t& layer) const {
  int signedlayer = static_cast<int>(SiStripRegionCabling::physicalLayer(subdet,layer));
  return (nlayers_ == -1 || signedlayer < nlayers_) ? true : false;
}

void SiStripRawToClustersRoI::random(RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {
  
  uint32_t total = cabling_->getRegionCabling().size();
  uint32_t required = static_cast<uint32_t>(CLHEP::RandFlat::shoot()*(total+1));
  for (uint32_t iregion = 0; iregion < required; iregion++) {
    for (uint32_t isubdet = 0; isubdet < SiStripRegionCabling::ALLSUBDETS; isubdet++) {
      SubDet subdet = static_cast<SubDet>(isubdet);
      for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
	if (!physicalLayer(subdet,ilayer)) break;
	cabling_->updateSiStripRefGetter<SiStripCluster>(refgetter,lazygetter,SiStripRegionCabling::elementIndex(iregion,subdet,ilayer));
      }
    }
  }
}

void SiStripRawToClustersRoI::electrons(const reco::SuperClusterCollection& collection, RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {
  
  reco::SuperClusterCollection::const_iterator icollection = collection.begin();
  for (; icollection!=collection.end(); icollection++) {
    Position position(icollection->eta(),icollection->phi());
    for (uint32_t isubdet = 0; isubdet < SiStripRegionCabling::ALLSUBDETS; isubdet++) {
      SubDet subdet = static_cast<SubDet>(isubdet);
      for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
	if (!physicalLayer(subdet,ilayer)) break;
	cabling_->updateSiStripRefGetter<SiStripCluster>(refgetter,lazygetter,position,electrondeta_,electrondphi_,subdet,ilayer);
      }
    }    
  }
}

void SiStripRawToClustersRoI::muons(const reco::TrackCollection& collection, RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {

  reco::TrackCollection::const_iterator icollection = collection.begin();
  for (; icollection!=collection.end(); icollection++) {
    Position position(icollection->outerPosition().eta(),icollection->outerPosition().phi());
    for (uint32_t isubdet = 0; isubdet < SiStripRegionCabling::ALLSUBDETS; isubdet++) {
      SubDet subdet = static_cast<SubDet>(isubdet);
      for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
	if (!physicalLayer(subdet,ilayer)) break;
	cabling_->updateSiStripRefGetter<SiStripCluster>(refgetter,lazygetter,position,muondeta_,muondphi_,subdet,ilayer);
      }
    }    
  }
}

void SiStripRawToClustersRoI::taujets(const reco::CaloJetCollection& collection, RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {

  reco::CaloJetCollection::const_iterator icollection = collection.begin();
  for (; icollection!=collection.end(); icollection++) {
    Position position(icollection->eta(),icollection->phi());
    for (uint32_t isubdet = 0; isubdet < SiStripRegionCabling::ALLSUBDETS; isubdet++) {
      SubDet subdet = static_cast<SubDet>(isubdet);
      for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
	if (!physicalLayer(subdet,ilayer)) break;
	cabling_->updateSiStripRefGetter<SiStripCluster>(refgetter,lazygetter,position,taujetdeta_,taujetdphi_,subdet,ilayer);
      }
    }    
  }
}

void SiStripRawToClustersRoI::bjets(const reco::CaloJetCollection& collection, RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {

  reco::CaloJetCollection::const_iterator icollection = collection.begin();
  for (; icollection!=collection.end(); icollection++) {
    Position position(icollection->eta(),icollection->phi());
    for (uint32_t isubdet = 0; isubdet < SiStripRegionCabling::ALLSUBDETS; isubdet++) {
      SubDet subdet = static_cast<SubDet>(isubdet);
      for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
	if (!physicalLayer(subdet,ilayer)) break;
	cabling_->updateSiStripRefGetter<SiStripCluster>(refgetter,lazygetter,position,bjetdeta_,bjetdphi_,subdet,ilayer);
      }
    }    
  }
}

