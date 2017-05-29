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
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Common/interface/Association.h"
//Main File
#include "fastjet/PseudoJet.hh"
#include "CommonTools/PileupAlgos/plugins/PuppiProducer.h"



// ------------------------------------------------------------------------------------------
PuppiProducer::PuppiProducer(const edm::ParameterSet& iConfig) {
  fPuppiDiagnostics = iConfig.getParameter<bool>("puppiDiagnostics");
  fPuppiForLeptons = iConfig.getParameter<bool>("puppiForLeptons");
  fUseDZ     = iConfig.getParameter<bool>("UseDeltaZCut");
  fDZCut     = iConfig.getParameter<double>("DeltaZCut");
  fUseExistingWeights     = iConfig.getParameter<bool>("useExistingWeights");
  fUseWeightsNoLep        = iConfig.getParameter<bool>("useWeightsNoLep");
  fClonePackedCands       = iConfig.getParameter<bool>("clonePackedCands");
  fVtxNdofCut = iConfig.getParameter<int>("vtxNdofCut");
  fVtxZCut = iConfig.getParameter<double>("vtxZCut");
  fPuppiContainer = std::unique_ptr<PuppiContainer> ( new PuppiContainer(iConfig) );

  tokenPFCandidates_
    = consumes<CandidateView>(iConfig.getParameter<edm::InputTag>("candName"));
  tokenVertices_
    = consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexName"));
 

  produces<edm::ValueMap<float> > ();
  produces<edm::ValueMap<LorentzVector> > ();
  produces< edm::ValueMap<reco::CandidatePtr> >(); 
  
  if (fUseExistingWeights || fClonePackedCands)
    produces<pat::PackedCandidateCollection>();
  else
    produces<reco::PFCandidateCollection>();

  if (fPuppiDiagnostics){
    produces<double> ("PuppiNAlgos");
    produces<std::vector<double>> ("PuppiRawAlphas");
    produces<std::vector<double>> ("PuppiAlphas");
    produces<std::vector<double>> ("PuppiAlphasMed");
    produces<std::vector<double>> ("PuppiAlphasRms");
  }
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

   int npv = 0;
   const reco::VertexCollection::const_iterator vtxEnd = pvCol->end();
   for (reco::VertexCollection::const_iterator vtxIter = pvCol->begin(); vtxEnd != vtxIter; ++vtxIter) {
      if (!vtxIter->isFake() && vtxIter->ndof()>=fVtxNdofCut && fabs(vtxIter->z())<=fVtxZCut)
         npv++;
   }

  //Fill the reco objects
  fRecoObjCollection.clear();
  for(CandidateView::const_iterator itPF = pfCol->begin(); itPF!=pfCol->end(); itPF++) {
    // std::cout << "itPF->pdgId() = " << itPF->pdgId() << std::endl;
    RecoObj pReco;
    pReco.pt  = itPF->pt();
    pReco.eta = itPF->eta();
    pReco.phi = itPF->phi();
    pReco.m   = itPF->mass();
    pReco.rapidity = itPF->rapidity();
    pReco.charge = itPF->charge(); 
    const reco::Vertex *closestVtx = 0;
    double pDZ    = -9999; 
    double pD0    = -9999; 
    int    pVtxId = -9999; 
    bool lFirst = true;
    const pat::PackedCandidate *lPack = dynamic_cast<const pat::PackedCandidate*>(&(*itPF));
    if(lPack == 0 ) {

      const reco::PFCandidate *pPF = dynamic_cast<const reco::PFCandidate*>(&(*itPF));
      double curdz = 9999;
      int closestVtxForUnassociateds = -9999;
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
        // in case it's unassocciated, keep more info
        double tmpdz = 99999;
        if      ( pPF->trackRef().isNonnull()    ) tmpdz = pPF->trackRef()   ->dz(iV->position());
        else if ( pPF->gsfTrackRef().isNonnull() ) tmpdz = pPF->gsfTrackRef()->dz(iV->position());
        if (fabs(tmpdz) < curdz){
          curdz = fabs(tmpdz);
          closestVtxForUnassociateds = pVtxId;
        }
        pVtxId++;

      }
      int tmpFromPV = 0;  
      // mocking the miniAOD definitions
      if (closestVtx != 0 && fabs(pReco.charge) > 0 && pVtxId > 0) tmpFromPV = 0;
      if (closestVtx != 0 && fabs(pReco.charge) > 0 && pVtxId == 0) tmpFromPV = 3;
      if (closestVtx == 0 && fabs(pReco.charge) > 0 && closestVtxForUnassociateds == 0) tmpFromPV = 2;
      if (closestVtx == 0 && fabs(pReco.charge) > 0 && closestVtxForUnassociateds != 0) tmpFromPV = 1;
      pReco.dZ      = pDZ;
      pReco.d0      = pD0;
      pReco.id = 0; 
      if (fabs(pReco.charge) == 0){ pReco.id = 0; }
      else{
        if (tmpFromPV == 0){ pReco.id = 2; } // 0 is associated to PU vertex
        if (tmpFromPV == 3){ pReco.id = 1; }
        if (tmpFromPV == 1 || tmpFromPV == 2){ 
          pReco.id = 0;
          if (!fPuppiForLeptons && fUseDZ && (fabs(pDZ) < fDZCut)) pReco.id = 1;
          if (!fPuppiForLeptons && fUseDZ && (fabs(pDZ) > fDZCut)) pReco.id = 2;
          if (fPuppiForLeptons && tmpFromPV == 1) pReco.id = 2;
          if (fPuppiForLeptons && tmpFromPV == 2) pReco.id = 1;
        }
      }
    } 
    else if(lPack->vertexRef().isNonnull() )  {
      pDZ        = lPack->dz();
      pD0        = lPack->dxy();
      closestVtx = &(*(lPack->vertexRef()));
      pReco.dZ      = pDZ;
      pReco.d0      = pD0;
  
      pReco.id = 0; 
      if (fabs(pReco.charge) == 0){ pReco.id = 0; }
      if (fabs(pReco.charge) > 0){
        if (lPack->fromPV() == 0){ pReco.id = 2; } // 0 is associated to PU vertex
        if (lPack->fromPV() == (pat::PackedCandidate::PVUsedInFit)){ pReco.id = 1; }
        if (lPack->fromPV() == (pat::PackedCandidate::PVTight) || lPack->fromPV() == (pat::PackedCandidate::PVLoose)){ 
          pReco.id = 0;
          if (!fPuppiForLeptons && fUseDZ && (fabs(pDZ) < fDZCut)) pReco.id = 1;
          if (!fPuppiForLeptons && fUseDZ && (fabs(pDZ) > fDZCut)) pReco.id = 2;
          if (fPuppiForLeptons && lPack->fromPV() == (pat::PackedCandidate::PVLoose)) pReco.id = 2;
          if (fPuppiForLeptons && lPack->fromPV() == (pat::PackedCandidate::PVTight)) pReco.id = 1;
        }
      }
    }

    fRecoObjCollection.push_back(pReco);
      
  }

  fPuppiContainer->initialize(fRecoObjCollection);
  fPuppiContainer->setNPV( npv );

  std::vector<double> lWeights;
  std::vector<fastjet::PseudoJet> lCandidates;
  if (!fUseExistingWeights){
    //Compute the weights and get the particles
    lWeights = fPuppiContainer->puppiWeights();
    lCandidates = fPuppiContainer->puppiParticles();
  }
  else{
    //Use the existing weights
    int lPackCtr = 0;
    for(CandidateView::const_iterator itPF = pfCol->begin(); itPF!=pfCol->end(); itPF++) {  
      const pat::PackedCandidate *lPack = dynamic_cast<const pat::PackedCandidate*>(&(*itPF));
      float curpupweight = -1.;
      if(lPack == 0 ) { 
        // throw error
        throw edm::Exception(edm::errors::LogicError,"PuppiProducer: cannot get weights since inputs are not PackedCandidates");
      }
      else{
        if (fUseWeightsNoLep){ curpupweight = lPack->puppiWeightNoLep(); }
        else{ curpupweight = lPack->puppiWeight();  }
      }
      lWeights.push_back(curpupweight);
      fastjet::PseudoJet curjet( curpupweight*lPack->px(), curpupweight*lPack->py(), curpupweight*lPack->pz(), curpupweight*lPack->energy());
      curjet.set_user_index(lPackCtr);
      lCandidates.push_back(curjet);
      lPackCtr++;
    }
  }

  //Fill it into the event
  std::auto_ptr<edm::ValueMap<float> > lPupOut(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler  lPupFiller(*lPupOut);
  lPupFiller.insert(hPFProduct,lWeights.begin(),lWeights.end());
  lPupFiller.fill();

  // This is a dummy to access the "translate" method which is a
  // non-static member function even though it doesn't need to be. 
  // Will fix in the future. 
  static const reco::PFCandidate dummySinceTranslateIsNotStatic;

  // Fill a new PF/Packed Candidate Collection and write out the ValueMap of the new p4s.
  // Since the size of the ValueMap must be equal to the input collection, we need
  // to search the "puppi" particles to find a match for each input. If none is found,
  // the input is set to have a four-vector of 0,0,0,0
  fPuppiCandidates.reset( new PFOutputCollection );
  fPackedPuppiCandidates.reset( new PackedOutputCollection );
  std::auto_ptr<edm::ValueMap<LorentzVector> > p4PupOut(new edm::ValueMap<LorentzVector>());
  LorentzVectorCollection puppiP4s;
  std::vector<reco::CandidatePtr> values(hPFProduct->size());

  for ( auto i0 = hPFProduct->begin(),
	  i0begin = hPFProduct->begin(),
	  i0end = hPFProduct->end(); i0 != i0end; ++i0 ) {
    std::unique_ptr<pat::PackedCandidate> pCand;
    std::unique_ptr<reco::PFCandidate>    pfCand;
    if (fUseExistingWeights || fClonePackedCands) {
      const pat::PackedCandidate *cand = dynamic_cast<const pat::PackedCandidate*>(&(*i0));
      if(!cand)
        throw edm::Exception(edm::errors::LogicError,"PuppiProducer: inputs are not PackedCandidates");
      pCand.reset( new pat::PackedCandidate(*cand) );
    } else {
      auto id = dummySinceTranslateIsNotStatic.translatePdgIdToType(i0->pdgId());
      const reco::PFCandidate *cand = dynamic_cast<const reco::PFCandidate*>(&(*i0));
      pfCand.reset( new reco::PFCandidate( cand ? *cand : reco::PFCandidate(i0->charge(), i0->p4(), id) ) );
    }
    LorentzVector pVec = i0->p4();
    int val = i0 - i0begin;

    // Find the Puppi particle matched to the input collection using the "user_index" of the object. 
    auto puppiMatched = find_if( lCandidates.begin(), lCandidates.end(), [&val]( fastjet::PseudoJet const & i ){ return i.user_index() == val; } );
    if ( puppiMatched != lCandidates.end() ) {
      pVec.SetPxPyPzE(puppiMatched->px(),puppiMatched->py(),puppiMatched->pz(),puppiMatched->E());
    } else {
      pVec.SetPxPyPzE( 0, 0, 0, 0);
    }
    puppiP4s.push_back( pVec );

    if (fUseExistingWeights || fClonePackedCands) {
      pCand->setP4(pVec);
      pCand->setSourceCandidatePtr( i0->sourceCandidatePtr(0) );
      fPackedPuppiCandidates->push_back(*pCand);
    } else {
      pfCand->setP4(pVec);
      pfCand->setSourceCandidatePtr( i0->sourceCandidatePtr(0) );
      fPuppiCandidates->push_back(*pfCand);
    }
  }

  //Compute the modified p4s
  edm::ValueMap<LorentzVector>::Filler  p4PupFiller(*p4PupOut);
  p4PupFiller.insert(hPFProduct,puppiP4s.begin(), puppiP4s.end() );
  p4PupFiller.fill();
  
  iEvent.put(lPupOut);
  iEvent.put(p4PupOut);
  if (fUseExistingWeights || fClonePackedCands) {
    edm::OrphanHandle<pat::PackedCandidateCollection> oh = iEvent.put( fPackedPuppiCandidates );
    for(unsigned int ic=0, nc = oh->size(); ic < nc; ++ic) {
        reco::CandidatePtr pkref( oh, ic );
        values[ic] = pkref;
    }
  } else {
    edm::OrphanHandle<reco::PFCandidateCollection> oh = iEvent.put( fPuppiCandidates );
    for(unsigned int ic=0, nc = oh->size(); ic < nc; ++ic) {
        reco::CandidatePtr pkref( oh, ic );
        values[ic] = pkref;
    }
  }
  std::auto_ptr<edm::ValueMap<reco::CandidatePtr> > pfMap_p(new edm::ValueMap<reco::CandidatePtr>());
  edm::ValueMap<reco::CandidatePtr>::Filler filler(*pfMap_p);
  filler.insert(hPFProduct, values.begin(), values.end());
  filler.fill();
  iEvent.put(pfMap_p);


  //////////////////////////////////////////////
  if (fPuppiDiagnostics && !fUseExistingWeights){

    // all the different alphas per particle
    // THE alpha per particle
    std::auto_ptr<std::vector<double> > theAlphas(new std::vector<double>(fPuppiContainer->puppiAlphas()));
    std::auto_ptr<std::vector<double> > theAlphasMed(new std::vector<double>(fPuppiContainer->puppiAlphasMed()));
    std::auto_ptr<std::vector<double> > theAlphasRms(new std::vector<double>(fPuppiContainer->puppiAlphasRMS()));
    std::auto_ptr<std::vector<double> > alphas(new std::vector<double>(fPuppiContainer->puppiRawAlphas()));
    std::auto_ptr<double> nalgos(new double(fPuppiContainer->puppiNAlgos()));
    
    iEvent.put(alphas,"PuppiRawAlphas");
    iEvent.put(nalgos,"PuppiNAlgos");
    iEvent.put(theAlphas,"PuppiAlphas");
    iEvent.put(theAlphasMed,"PuppiAlphasMed");
    iEvent.put(theAlphasRms,"PuppiAlphasRms");
  }
  
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
