// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Math/interface/deltaR.h"
//Main File
#include "CommonTools/PileupAlgos/plugins/PuppiPhoton.h"

// ------------------------------------------------------------------------------------------
PuppiPhoton::PuppiPhoton(const edm::ParameterSet& iConfig) {
  tokenPFCandidates_     = consumes<CandidateView>(iConfig.getParameter<edm::InputTag>("candName"));
  tokenPuppiCandidates_  = consumes<CandidateView>(iConfig.getParameter<edm::InputTag>("puppiCandName"));
  tokenPhotonCandidates_ = consumes<CandidateView>(iConfig.getParameter<edm::InputTag>("photonName"));
  reco2pf_               = mayConsume<edm::ValueMap<std::vector<reco::PFCandidateRef> > >(iConfig.getParameter<edm::InputTag>("recoToPFMap"));
  tokenPhotonId_         = mayConsume<edm::ValueMap<bool>  >(iConfig.getParameter<edm::InputTag>("photonId"));
  pt_                    = iConfig.getParameter<double>("pt");
  eta_                   = iConfig.getParameter<double>("eta");
  dRMatch_               = iConfig.getParameter<std::vector<double> > ("dRMatch");
  pdgIds_                = iConfig.getParameter<std::vector<int32_t> >("pdgids");
  runOnMiniAOD_          = iConfig.getParameter<bool>("runOnMiniAOD");
  usePFRef_              = iConfig.getParameter<bool>("useRefs");
  weight_                = iConfig.getParameter<double>("weight");
  useValueMap_           = iConfig.getParameter<bool>("useValueMap");
  tokenWeights_          = consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("weightsName"));

  usePhotonId_  = (iConfig.getParameter<edm::InputTag>("photonId")).label().size() != 0;
  produces<PFOutputCollection>();
  produces< edm::ValueMap<reco::CandidatePtr> >(); 
}
// ------------------------------------------------------------------------------------------
PuppiPhoton::~PuppiPhoton(){}
// ------------------------------------------------------------------------------------------
void PuppiPhoton::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  edm::Handle<CandidateView> hPhoProduct;
  iEvent.getByToken(tokenPhotonCandidates_,hPhoProduct);
  const CandidateView *phoCol = hPhoProduct.product();

  edm::Handle<edm::ValueMap<bool> > photonId;
  if(usePhotonId_) iEvent.getByToken(tokenPhotonId_,photonId);
  int iC = -1;
  std::vector<const reco::Candidate*> phoCands;
  std::vector<uint16_t> phoIndx;

  edm::Handle<edm::ValueMap<std::vector<reco::PFCandidateRef>>> reco2pf;
  if(!runOnMiniAOD_) iEvent.getByToken(reco2pf_, reco2pf);

  // Get PFCandidate Collection
  edm::Handle<CandidateView> hPFProduct;
  iEvent.getByToken(tokenPFCandidates_,hPFProduct);
  const CandidateView *pfCol = hPFProduct.product();

  edm::Handle<CandidateView> hPuppiProduct;
  iEvent.getByToken(tokenPuppiCandidates_,hPuppiProduct);
  const CandidateView *pupCol = hPuppiProduct.product();
  for(CandidateView::const_iterator itPho = phoCol->begin(); itPho!=phoCol->end(); itPho++) {
    iC++;
    bool passObject = false;
    if(itPho->isPhoton() && usePhotonId_)   passObject =  (*photonId)  [phoCol->ptrAt(iC)];
    if(itPho->pt() < pt_) continue;
    if(!passObject && usePhotonId_) continue;
    if(runOnMiniAOD_) {
      const pat::Photon *pPho = dynamic_cast<const pat::Photon*>(&(*itPho));
      if(pPho != 0) {
        for( const edm::Ref<pat::PackedCandidateCollection> & ref : pPho->associatedPackedPFCandidates() ) {
	  if(fabs(pfCol->ptrAt(ref.key())->eta()) < eta_ ) {
	    phoIndx.push_back(ref.key());
	    phoCands.push_back(&(*(pfCol->ptrAt(ref.key()))));
	  }
        }
        continue;
      }
      const pat::Electron *pElectron = dynamic_cast<const pat::Electron*>(&(*itPho));
      if(pElectron != 0) {
        for( const edm::Ref<pat::PackedCandidateCollection> & ref : pElectron->associatedPackedPFCandidates() )
	  if(fabs(pfCol->ptrAt(ref.key())->eta()) < eta_ )  {
	    phoIndx.push_back(ref.key());
	    phoCands.push_back(&(*(pfCol->ptrAt(ref.key()))));
	  }
      }
    } else {
      for( const edm::Ref<std::vector<reco::PFCandidate> > & ref : (*reco2pf)[phoCol->ptrAt(iC)] ) {
	  if(fabs(pfCol->ptrAt(ref.key())->eta()) < eta_ )  {
	    phoIndx.push_back(ref.key());
	    phoCands.push_back(&(*(pfCol->ptrAt(ref.key()))));
	  }
      }
    }
  }
  //Get Weights
  edm::Handle<edm::ValueMap<float> > pupWeights; 
  iEvent.getByToken(tokenWeights_,pupWeights);
  std::auto_ptr<edm::ValueMap<LorentzVector> > p4PupOut(new edm::ValueMap<LorentzVector>());
  LorentzVectorCollection puppiP4s;
  std::vector<reco::CandidatePtr> values(hPFProduct->size());
  int iPF = 0; 
  std::vector<float> lWeights;
  static const reco::PFCandidate dummySinceTranslateIsNotStatic;
  corrCandidates_.reset( new PFOutputCollection );
  std::set<int> foundPhoIndex;
  for(CandidateView::const_iterator itPF = pupCol->begin(); itPF!=pupCol->end(); itPF++) {  
    auto id = dummySinceTranslateIsNotStatic.translatePdgIdToType(itPF->pdgId());
    const reco::PFCandidate *pPF = dynamic_cast<const reco::PFCandidate*>(&(*itPF));
    reco::PFCandidate pCand( pPF ? *pPF : reco::PFCandidate(itPF->charge(), itPF->p4(), id) );
    LorentzVector pVec = itPF->p4();
    float pWeight = 1.;
    if(useValueMap_) pWeight  = (*pupWeights)[pupCol->ptrAt(iPF)];     
    if(!usePFRef_) { 
      int iPho = -1;
      for(std::vector<const reco::Candidate*>::iterator itPho = phoCands.begin(); itPho!=phoCands.end(); itPho++) {
	iPho++;
	if((!matchPFCandidate(&(*itPF),*itPho))||(foundPhoIndex.count(iPho)!=0)) continue;
        pWeight = weight_;
	if(!useValueMap_ && itPF->pt() != 0) pWeight = pWeight*(phoCands[iPho]->pt()/itPF->pt());
	if(!useValueMap_ && itPF->pt() == 0) pVec.SetPxPyPzE(phoCands[iPho]->px()*pWeight,phoCands[iPho]->py()*pWeight,phoCands[iPho]->pz()*pWeight,phoCands[iPho]->energy()*pWeight);
        foundPhoIndex.insert(iPho);
      }
    } else { 
      int iPho = -1;
      for(std::vector<uint16_t>::const_iterator itPho = phoIndx.begin(); itPho!=phoIndx.end(); itPho++) {
        iPho++;
        if(pupCol->refAt(iPF).key() != *itPho) continue;
        pWeight = weight_;
        if(!useValueMap_ && itPF->pt() != 0) pWeight = pWeight*(phoCands[iPho]->pt()/itPF->pt());
	if(!useValueMap_ && itPF->pt() == 0) pVec.SetPxPyPzE(phoCands[iPho]->px()*pWeight,phoCands[iPho]->py()*pWeight,phoCands[iPho]->pz()*pWeight,phoCands[iPho]->energy()*pWeight);
        foundPhoIndex.insert(iPho);
      }
    }
    if(itPF->pt() != 0) pVec.SetPxPyPzE(itPF->px()*pWeight,itPF->py()*pWeight,itPF->pz()*pWeight,itPF->energy()*pWeight);

    lWeights.push_back(pWeight);
    pCand.setP4(pVec);
    puppiP4s.push_back( pVec );
    pCand.setSourceCandidatePtr( itPF->sourceCandidatePtr(0) );
    corrCandidates_->push_back(pCand);
    iPF++;
  }
  //Add the missing pfcandidates
  for(unsigned int iPho = 0; iPho < phoCands.size(); iPho++) { 
    if(foundPhoIndex.count(iPho)!=0) continue;
    auto id = dummySinceTranslateIsNotStatic.translatePdgIdToType(phoCands[iPho]->pdgId());
    reco::PFCandidate pCand(reco::PFCandidate(phoCands[iPho]->charge(), phoCands[iPho]->p4(),id) );
    pCand.setSourceCandidatePtr( phoCands[iPho]->sourceCandidatePtr(0) );
    LorentzVector pVec = phoCands[iPho]->p4();
    pVec.SetPxPyPzE(phoCands[iPho]->px()*weight_,phoCands[iPho]->py()*weight_,phoCands[iPho]->pz()*weight_,phoCands[iPho]->energy()*weight_);
    pCand.setP4(pVec);
    lWeights.push_back(weight_);
    puppiP4s.push_back( pVec );
    corrCandidates_->push_back(pCand);
  }
  //Fill it into the event
  edm::OrphanHandle<reco::PFCandidateCollection> oh = iEvent.put( corrCandidates_ );
  for(unsigned int ic=0, nc = pupCol->size(); ic < nc; ++ic) {
      reco::CandidatePtr pkref( oh, ic );
      values[ic] = pkref;
  }  
  std::auto_ptr<edm::ValueMap<reco::CandidatePtr> > pfMap_p(new edm::ValueMap<reco::CandidatePtr>());
  edm::ValueMap<reco::CandidatePtr>::Filler filler(*pfMap_p);
  filler.insert(hPFProduct, values.begin(), values.end());
  filler.fill();
  iEvent.put(pfMap_p);
}
// ------------------------------------------------------------------------------------------
bool PuppiPhoton::matchPFCandidate(const reco::Candidate *iPF,const reco::Candidate *iPho) { 
  double lDR = deltaR(iPF->eta(),iPF->phi(),iPho->eta(),iPho->phi());
  for(unsigned int i0 = 0; i0 < pdgIds_.size(); i0++) {
    if(std::abs(iPF->pdgId()) == pdgIds_[i0] && lDR < dRMatch_[i0])  return true;
  }
  return false;
}
// ------------------------------------------------------------------------------------------
void PuppiPhoton::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(PuppiPhoton);
