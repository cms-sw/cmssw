#include "Calibration/HcalIsolatedTrackReco/interface/HITSiStripRawToClustersRoI.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h" 
#include "CLHEP/Random/RandFlat.h"


using namespace std;
using namespace sistrip;

HITSiStripRawToClustersRoI::HITSiStripRawToClustersRoI(const edm::ParameterSet& conf) :

  cabling_(),
  allregions_(),
  nlayers_(conf.getUntrackedParameter<int>("layers",-1)),
  global_(conf.getUntrackedParameter<bool>("doGlobal",true)),
  random_(conf.getUntrackedParameter<bool>("random",false)),
  taujets_(conf.getUntrackedParameter<bool>("useTauJets",false)),
  ptrack_(conf.getUntrackedParameter<bool>("usePixelTracks",true)),
  taujetdeta_(conf.getUntrackedParameter<double>("tjetEtaWindow",0.2)),
  taujetdphi_(conf.getUntrackedParameter<double>("tjetPhiWindow",0.2)),
  ptrackEta_(conf.getUntrackedParameter<double>("ptrackEtaWindow",0.3)),
  ptrackPhi_(conf.getUntrackedParameter<double>("ptrackPhiWindow",0.3))
	   
{
  tok_siStrip_ = consumes<LazyGetter>(conf.getParameter<edm::InputTag>("siStripLazyGetter"));
  tok_tauL1_ = consumes<l1extra::L1JetParticleCollection>(conf.getParameter<edm::InputTag>("l1tauJetLabel"));
  tok_ptrack_ = consumes<trigger::TriggerFilterObjectWithRefs>(conf.getParameter<edm::InputTag>("pixelTrackLabel"));

  produces< RefGetter >();
}

HITSiStripRawToClustersRoI::~HITSiStripRawToClustersRoI() {}

void HITSiStripRawToClustersRoI::beginJob() {}

void HITSiStripRawToClustersRoI::endJob() {}

void HITSiStripRawToClustersRoI::produce(edm::Event& event, const edm::EventSetup& setup) {

  setup.get<SiStripRegionCablingRcd>().get(cabling_);
  allregions_.reserve(cabling_->getRegionCabling().size());
  for (uint32_t iregion=0;iregion<cabling_->getRegionCabling().size();iregion++) {
    for (uint32_t isubdet=0;isubdet<cabling_->getRegionCabling()[iregion].size();isubdet++) {
      for (uint32_t ilayer=0;ilayer<cabling_->getRegionCabling()[iregion][isubdet].size();ilayer++) {
        uint32_t index = SiStripRegionCabling::elementIndex(iregion,static_cast<SubDet>(isubdet),ilayer);
        allregions_.push_back(index);
      }
    }
  }

  
  edm::Handle< LazyGetter > lazygetter;
  event.getByToken(tok_siStrip_,lazygetter);

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
  
  /// Seeded by L1 taujets.

  if (taujets_) {
	edm::Handle<l1extra::L1JetParticleCollection> collection;
	event.getByToken(tok_tauL1_,collection);
	taujets(*collection,*refgetter,lazygetter);
    }


  ///seeded by isol tracks
  if (ptrack_) {
      edm::Handle<trigger::TriggerFilterObjectWithRefs> collection;
      event.getByToken(tok_ptrack_,collection);
      ptracks(*collection,*refgetter,lazygetter);
    }
  
  event.put(refgetter);
}

bool HITSiStripRawToClustersRoI::physicalLayer(SubDet& subdet, uint32_t& layer) const {
  uint32_t signedlayer = 0;
  if (subdet == SiStripRegionCabling::TIB || subdet == SiStripRegionCabling::TID) signedlayer = layer;
  else if (subdet == SiStripRegionCabling::TOB) signedlayer = SiStripRegionCabling::TIBLAYERS + layer;
  else if (subdet == SiStripRegionCabling::TEC) signedlayer = SiStripRegionCabling::TIDLAYERS + layer;
  else signedlayer = SiStripRegionCabling::ALLLAYERS;

  return (nlayers_ == -1 || int(signedlayer) < nlayers_) ? true : false;
}

void HITSiStripRawToClustersRoI::random(RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {
  
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


void HITSiStripRawToClustersRoI::taujets(const l1extra::L1JetParticleCollection& collection, RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {

  l1extra::L1JetParticleCollection::const_iterator icollection = collection.begin();
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


void HITSiStripRawToClustersRoI::ptracks(const trigger::TriggerFilterObjectWithRefs& collection, RefGetter& refgetter, edm::Handle<LazyGetter>& lazygetter) const {

  std::vector< edm::Ref<reco::IsolatedPixelTrackCandidateCollection> > isoPixTrackRefs;
  collection.getObjects(trigger::TriggerTrack, isoPixTrackRefs);
  for (uint32_t p=0 ; p<isoPixTrackRefs.size(); p++) {
    Position position(isoPixTrackRefs[p]->track()->eta(),isoPixTrackRefs[p]->track()->phi());
    for (uint32_t isubdet = 0; isubdet < SiStripRegionCabling::ALLSUBDETS; isubdet++) {
      SubDet subdet = static_cast<SubDet>(isubdet);
      for (uint32_t ilayer = 0; ilayer < SiStripRegionCabling::ALLLAYERS; ilayer++) {
	if (!physicalLayer(subdet,ilayer)) break;
	cabling_->updateSiStripRefGetter<SiStripCluster>(refgetter,lazygetter,position,ptrackEta_,ptrackPhi_,subdet,ilayer);
      }
    }
    
  }
}
