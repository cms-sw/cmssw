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
 #include "CommonTools/Puppi/plugins/PuppiProducer.h"

typedef edm::View<reco::Candidate> CandidateView;

 // ------------------------------------------------------------------------------------------
PuppiProducer::PuppiProducer(const edm::ParameterSet& iConfig) {
 fPuppiName = iConfig.getUntrackedParameter<std::string>("PuppiName");
 fUseDZ     = iConfig.getUntrackedParameter<bool>("UseDeltaZCut");
 fDZCut     = iConfig.getUntrackedParameter<double>("DeltaZCut");
 fPuppiContainer = new PuppiContainer(iConfig);
 fPFName    = iConfig.getUntrackedParameter<std::string>("candName"  ,"particleFlow");
 fPVName    = iConfig.getUntrackedParameter<std::string>("vertexName","offlinePrimaryVertices");
 produces<edm::ValueMap<float> > ("PuppiWeights");
 produces<reco::PFCandidateCollection>(fPuppiName);
}
 // ------------------------------------------------------------------------------------------
PuppiProducer::~PuppiProducer(){
}
 // ------------------------------------------------------------------------------------------
void PuppiProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

   // Get PFCandidate Collection
 edm::Handle<CandidateView> hPFProduct;
 iEvent.getByLabel(fPFName,hPFProduct);
 assert(hPFProduct.isValid());
 const reco::CandidateView *PFCol = hPFProduct.product();

   // Get vertex collection w/PV as the first entry?
 edm::Handle<reco::VertexCollection> hVertexProduct;
 iEvent.getByLabel(fPVName,hVertexProduct);
 assert(hVertexProduct.isValid());
 const reco::VertexCollection *pvCol = hVertexProduct.product();

   //Fill the reco objects
 fRecoObjCollection.clear();
 for(CandidateView::const_iterator itPF = PFCol->begin(); itPF!=PFCol->end(); itPF++) {
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
   pVtxId     = (lPack->fromPV() !=  (pat::PackedCandidate::PVUsedInFit));
   if(lPack->fromPV() ==  (pat::PackedCandidate::PVLoose || pat::PackedCandidate::PVTight)) closestVtx = 0;
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
 if(fUseDZ && pDZ > -9999 && closestVtx == 0 && (fabs(pDZ) < fDZCut)) pReco.id = 1; 
 if(fUseDZ && pDZ > -9999 && closestVtx == 0 && (fabs(pDZ) > fDZCut)) pReco.id = 2; 

  //std::cout << "pVtxId = " << pVtxId << ", and charge = " << itPF->charge() << ", and closestVtx = " << closestVtx << ", and id = " << pReco.id << std::endl;

 fRecoObjCollection.push_back(pReco);
}
fPuppiContainer->initialize(fRecoObjCollection);

   //Compute the weights and the candidates
const std::vector<double> lWeights = fPuppiContainer->puppiWeights();
   //Fill it into the event
std::auto_ptr<edm::ValueMap<float> > lPupOut(new edm::ValueMap<float>());
edm::ValueMap<float>::Filler  lPupFiller(*lPupOut);
lPupFiller.insert(hPFProduct,lWeights.begin(),lWeights.end());
lPupFiller.fill();  
iEvent.put(lPupOut,"PuppiWeights");
   //Fill a new PF Candidate Collection
const std::vector<fastjet::PseudoJet> lCandidates = fPuppiContainer->puppiParticles();
fPuppiCandidates.reset( new reco::PFCandidateCollection );    
for(unsigned int i0 = 0; i0 < lCandidates.size(); i0++) {
     //reco::PFCandidate pCand;
 reco::PFCandidate pCand(PFCol->at(lCandidates[i0].user_index()).charge(),
  PFCol->at(lCandidates[i0].user_index()).p4(),
  translatePdgIdToType(PFCol->at(lCandidates[i0].user_index()).pdgId()));
     LorentzVector pVec; //pVec.SetPtEtaPhiM(lCandidates[i0].pt(),lCandidates[i0].eta(),lCandidates[i0].phi(),lCandidates[i0].Mass());
     pVec.SetPxPyPzE(lCandidates[i0].px(),lCandidates[i0].py(),lCandidates[i0].pz(),lCandidates[i0].E());
     pCand.setP4(pVec);
     fPuppiCandidates->push_back(pCand);
   }
   iEvent.put(fPuppiCandidates,fPuppiName);
 }
// ------------------------------------------------------------------------------------------
 reco::PFCandidate::ParticleType PuppiProducer::translatePdgIdToType(int pdgid) const {
  switch (std::abs(pdgid)) {
    case 211: return reco::PFCandidate::h;
    case 11:  return reco::PFCandidate::e;
    case 13:  return reco::PFCandidate::mu;
    case 22:  return reco::PFCandidate::gamma;
    case 130: return reco::PFCandidate::h0;
    case 1:   return reco::PFCandidate::h_HF;
    case 2:   return reco::PFCandidate::egamma_HF;
    case 0:   return reco::PFCandidate::X;  
    default: return reco::PFCandidate::X;
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
