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
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
//Main File
#include "fastjet/PseudoJet.hh"
#include "CommonTools/PileupAlgos/plugins/PuppiProducer.h"



// ------------------------------------------------------------------------------------------
PuppiProducer::PuppiProducer(const edm::ParameterSet& iConfig) {
  fUseDZ     = iConfig.getParameter<bool>("UseDeltaZCut");
  fDZCut     = iConfig.getParameter<double>("DeltaZCut");
  fPuppiContainer = std::unique_ptr<PuppiContainer> ( new PuppiContainer(iConfig) );

  tokenPFCandidates_
    = consumes<CandidateView>(iConfig.getParameter<edm::InputTag>("candName"));
  tokenVertices_
    = consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexName"));
 

  produces<edm::ValueMap<float> > ("PuppiWeights");
  produces<edm::ValueMap<LorentzVector> > ("PuppiP4s");
  produces<PFOutputCollection>();


}
// ------------------------------------------------------------------------------------------
PuppiProducer::~PuppiProducer(){
}
// ------------------------------------------------------------------------------------------
void PuppiProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // Get PFCandidate Collection
  edm::Handle<CandidateView> hPFProduct;
  iEvent.getByToken(tokenPFCandidates_,hPFProduct);
  const CandidateView *pfCol = hPFProduct.product();

  // Get vertex collection w/PV as the first entry?
  edm::Handle<reco::VertexCollection> hVertexProduct;
  iEvent.getByToken(tokenVertices_,hVertexProduct);
  const reco::VertexCollection *pvCol = hVertexProduct.product();

  //Fill the reco objects
  fRecoObjCollection.clear();
  for(CandidateView::const_iterator itPF = pfCol->begin(); itPF!=pfCol->end(); itPF++) {
    RecoObj pReco;
    pReco.pt  = itPF->pt();
    pReco.eta = itPF->eta();
    pReco.phi = itPF->phi();
    pReco.m   = itPF->mass();
    pReco.charge = itPF->charge(); 
    const reco::Vertex *closestVtx = 0;
    double pDZ    = -9999; 
    double pD0    = -9999; 
    int    pVtxId = -9999; 
    bool lFirst = true;
    const pat::PackedCandidate *lPack = dynamic_cast<const pat::PackedCandidate*>(&(*itPF));
    if(lPack == 0 ) { 
      const reco::PFCandidate *pPF = dynamic_cast<const reco::PFCandidate*>(&(*itPF));
      for(reco::VertexCollection::const_iterator iV = pvCol->begin(); iV!=pvCol->end(); ++iV) {
	if(lFirst) { 
	  if      ( pPF->trackRef().isNonnull()    ) pDZ = pPF->trackRef()   ->dz(iV->position());
	  else if ( pPF->gsfTrackRef().isNonnull() ) pDZ = pPF->gsfTrackRef()->dz(iV->position());
	  if      ( pPF->trackRef().isNonnull()    ) pD0 = pPF->trackRef()   ->d0();
	  else if ( pPF->gsfTrackRef().isNonnull() ) pD0 = pPF->gsfTrackRef()->d0();
	  lFirst = false;
	  if(pDZ > -9999) pVtxId = 0; 
	}
	if(iV->trackWeight(pPF->trackRef())>0) {
	  closestVtx  = &(*iV);
	  break;
	}
	pVtxId++;
      }
    } else if(lPack->vertexRef().isNonnull() )  {
      pDZ        = lPack->dz(); 
      pD0        = lPack->dxy(); 
      closestVtx = &(*(lPack->vertexRef()));
      pVtxId = (lPack->fromPV() != (pat::PackedCandidate::PVUsedInFit)); 
      if( (lPack->fromPV() == pat::PackedCandidate::PVLoose) || 
	  (lPack->fromPV() == pat::PackedCandidate::PVTight) ) 
	closestVtx = 0; 
    }
    pReco.dZ      = pDZ;
    pReco.d0      = pD0;

    if(closestVtx == 0) pReco.vtxId = -1;
    if(closestVtx != 0) pReco.vtxId = pVtxId;
    //if(closestVtx != 0) pReco.vtxChi2 = closestVtx->trackWeight(itPF->trackRef());
    //Set the id for Puppi Algo: 0 is neutral pfCandidate, id = 1 for particles coming from PV and id = 2 for charged particles from non-leading vertex
    pReco.id       = 0; 

    if(closestVtx != 0 && pVtxId == 0 && fabs(pReco.charge) > 0) pReco.id = 1;
    if(closestVtx != 0 && pVtxId >  0 && fabs(pReco.charge) > 0) pReco.id = 2;
    //Add a dZ cut if wanted (this helps)
    if(fUseDZ && pDZ > -9999 && closestVtx == 0 && (fabs(pDZ) < fDZCut) && fabs(pReco.charge) > 0) pReco.id = 1; 
    if(fUseDZ && pDZ > -9999 && closestVtx == 0 && (fabs(pDZ) > fDZCut) && fabs(pReco.charge) > 0) pReco.id = 2; 

    //std::cout << "pVtxId = " << pVtxId << ", and charge = " << itPF->charge() << ", and closestVtx = " << closestVtx << ", and id = " << pReco.id << std::endl;

    fRecoObjCollection.push_back(pReco);
  }
  fPuppiContainer->initialize(fRecoObjCollection);

  //Compute the weights
  const std::vector<double> lWeights = fPuppiContainer->puppiWeights();
  //Fill it into the event
  std::auto_ptr<edm::ValueMap<float> > lPupOut(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler  lPupFiller(*lPupOut);
  lPupFiller.insert(hPFProduct,lWeights.begin(),lWeights.end());
  lPupFiller.fill();

  // This is a dummy to access the "translate" method which is a
  // non-static member function even though it doesn't need to be. 
  // Will fix in the future. 
  static const reco::PFCandidate dummySinceTranslateIsNotStatic;

  //Fill a new PF Candidate Collection and write out the ValueMap of the new p4s.
  // Since the size of the ValueMap must be equal to the input collection, we need
  // to search the "puppi" particles to find a match for each input. If none is found,
  // the input is set to have a four-vector of 0,0,0,0
  const std::vector<fastjet::PseudoJet> lCandidates = fPuppiContainer->puppiParticles();
  fPuppiCandidates.reset( new PFOutputCollection );
  std::auto_ptr<edm::ValueMap<LorentzVector> > p4PupOut(new edm::ValueMap<LorentzVector>());
  LorentzVectorCollection puppiP4s;
  for ( auto i0 = hPFProduct->begin(),
	  i0begin = hPFProduct->begin(),
	  i0end = hPFProduct->end(); i0 != i0end; ++i0 ) {
    //for(unsigned int i0 = 0; i0 < lCandidates.size(); i0++) {
    //reco::PFCandidate pCand;
    auto id = dummySinceTranslateIsNotStatic.translatePdgIdToType(i0->pdgId());
    reco::PFCandidate pCand( i0->charge(),
			     i0->p4(),
			     id );
    LorentzVector pVec = i0->p4();
    int val = i0 - i0begin;

    // Find the Puppi particle matched to the input collection using the "user_index" of the object. 
    auto puppiMatched = find_if( lCandidates.begin(), lCandidates.end(), [&val]( fastjet::PseudoJet const & i ){ return i.user_index() == val; } );
    if ( puppiMatched != lCandidates.end() ) {
      pVec.SetPxPyPzE(puppiMatched->px(),puppiMatched->py(),puppiMatched->pz(),puppiMatched->E());
    } else {
      pVec.SetPxPyPzE( 0, 0, 0, 0);
    }
    pCand.setP4(pVec);
    puppiP4s.push_back( pVec );
    fPuppiCandidates->push_back(pCand);
  }

  //Compute the modified p4s
  edm::ValueMap<LorentzVector>::Filler  p4PupFiller(*p4PupOut);
  p4PupFiller.insert(hPFProduct,puppiP4s.begin(), puppiP4s.end() );
  p4PupFiller.fill();

  iEvent.put(lPupOut,"PuppiWeights");
  iEvent.put(p4PupOut,"PuppiP4s");
  iEvent.put(fPuppiCandidates);
}

// ------------------------------------------------------------------------------------------
void PuppiProducer::beginJob() {
}
// ------------------------------------------------------------------------------------------
void PuppiProducer::endJob() {
}
// ------------------------------------------------------------------------------------------
void PuppiProducer::beginRun(edm::Run&, edm::EventSetup const&) {
}
// ------------------------------------------------------------------------------------------
void PuppiProducer::endRun(edm::Run&, edm::EventSetup const&) {
}
// ------------------------------------------------------------------------------------------
void PuppiProducer::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&) {
}
// ------------------------------------------------------------------------------------------
void PuppiProducer::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&) {
}
// ------------------------------------------------------------------------------------------
void PuppiProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(PuppiProducer);
