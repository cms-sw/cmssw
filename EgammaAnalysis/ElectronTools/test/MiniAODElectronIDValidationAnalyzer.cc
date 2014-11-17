// -*- C++ -*-
//
// Package:    TestElectronID/MiniAODElectronIDValidationAnalyzer
// Class:      MiniAODElectronIDValidationAnalyzer
// 
/**\class MiniAODElectronIDValidationAnalyzer MiniAODElectronIDValidationAnalyzer.cc TestElectronID/MiniAODElectronIDValidationAnalyzer/plugins/MiniAODElectronIDValidationAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Ilya Kravchenko
//         Created:  Thu, 14 Aug 2014 08:27:41 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TTree.h"
#include "Math/VectorUtil.h"

//
// class declaration
//

class MiniAODElectronIDValidationAnalyzer : public edm::EDAnalyzer {
   public:
      explicit MiniAODElectronIDValidationAnalyzer(const edm::ParameterSet&);
      ~MiniAODElectronIDValidationAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  enum ElectronMatchType {UNMATCHED = 0, 
			  TRUE_PROMPT_ELECTRON, 
			  TRUE_ELECTRON_FROM_TAU,
			  TRUE_NON_PROMPT_ELECTRON}; // The last does not include tau parents

   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  
      int matchToTruth(const reco::GsfElectron &el, const edm::Handle<edm::View<reco::GenParticle>>  &genParticles);
      void findFirstNonElectronMother(const reco::Candidate *particle, int &ancestorPID, int &ancestorStatus);
 
      // ----------member data ---------------------------
      edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
      edm::EDGetTokenT<edm::View<reco::GenParticle> > genToken_;
      edm::EDGetTokenT<reco::ConversionCollection> convToken_;
      edm::EDGetTokenT<reco::BeamSpot> beamToken_;
      edm::EDGetTokenT<edm::ValueMap<float> > full5x5SigmaIEtaIEtaMapToken_;
      edm::EDGetTokenT<edm::View<pat::Electron> > electronCollectionToken_;
      edm::InputTag electronIdTag_;
      edm::EDGetTokenT<edm::ValueMap<bool> > electronIdToken_;

      TTree *electronTree_;
      Float_t pt_;
      Float_t etaSC_;
  // All ID variables
  Float_t dEtaIn_;
  Float_t dPhiIn_;
  Float_t hOverE_;
  Float_t sigmaIetaIeta_;
  Float_t full5x5_sigmaIetaIeta_;
  Float_t relIsoWithDBeta_;
  Float_t ooEmooP_;
  Float_t d0_;
  Float_t dz_;
  Int_t   expectedMissingInnerHits_;
  Int_t   passConversionVeto_;     

      Int_t   isTrue_;
      Int_t   isPass_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MiniAODElectronIDValidationAnalyzer::MiniAODElectronIDValidationAnalyzer(const edm::ParameterSet& iConfig):
  vtxToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
  genToken_(consumes<edm::View<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("genparticles"))),
  convToken_(consumes<reco::ConversionCollection>(iConfig.getParameter<edm::InputTag>("convcollection"))),
  beamToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamspot"))),
  full5x5SigmaIEtaIEtaMapToken_(consumes<edm::ValueMap<float> >(iConfig.getParameter<edm::InputTag>("full5x5SigmaIEtaIEtaMap"))),
  electronCollectionToken_(consumes<edm::View<pat::Electron> >(iConfig.getParameter<edm::InputTag>("electrons"))),
  electronIdTag_(iConfig.getParameter<edm::InputTag>("electronIDs")),
  electronIdToken_(consumes<edm::ValueMap<bool> >(iConfig.getParameter<edm::InputTag>("electronIDs")))
{
   //now do what ever initialization is needed
  edm::Service<TFileService> fs;
  electronTree_ = fs->make<TTree> ("ElectronTree", "Electron data");
  
  electronTree_->Branch("pt"      ,  &pt_    , "pt/F");                       
  electronTree_->Branch("etaSC"   ,  &etaSC_ , "etaSC/F");

  electronTree_->Branch("dEtaIn",  &dEtaIn_, "dEtaIn/F");
  electronTree_->Branch("dPhiIn",  &dPhiIn_, "dPhiIn/F");
  electronTree_->Branch("hOverE",  &hOverE_, "hOverE/F");
  electronTree_->Branch("sigmaIetaIeta",         &sigmaIetaIeta_, "sigmaIetaIeta/F");
  electronTree_->Branch("full5x5_sigmaIetaIeta", &full5x5_sigmaIetaIeta_, "full5x5_sigmaIetaIeta/F");
  electronTree_->Branch("relIsoWithDBeta"      , &relIsoWithDBeta_, "relIsoWithDBeta/F");
  electronTree_->Branch("ooEmooP", &ooEmooP_, "ooEmooP/F");
  electronTree_->Branch("d0"     , &d0_,      "d0/F");
  electronTree_->Branch("dz"     , &dz_,      "dz/F");
  electronTree_->Branch("expectedMissingInnerHits", &expectedMissingInnerHits_, "expectedMissingInnerHits/I");
  electronTree_->Branch("passConversionVeto", &passConversionVeto_, "passConversionVeto/I");


  electronTree_->Branch("isTrue"  ,  &isTrue_ , "isTrue/I");
  electronTree_->Branch("isPass" ,  &isPass_ , "isPass/I");

}


MiniAODElectronIDValidationAnalyzer::~MiniAODElectronIDValidationAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
MiniAODElectronIDValidationAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // using namespace edm;

  //edm::Handle<edm::ValueMap<float> > full5x5sieie;
  edm::Handle<edm::View<pat::Electron> > collection;
  //edm::Handle<edm::ValueMap<bool> > id_decisions;
  //iEvent.getByToken(full5x5SigmaIEtaIEtaMapToken_,full5x5sieie);
  iEvent.getByToken(electronCollectionToken_, collection);
  //iEvent.getByToken(electronIdToken_,id_decisions);
  
  // Get PV
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vtxToken_, vertices);
  if (vertices->empty()) return; // skip the event if no PV found
  const reco::Vertex &pv = vertices->front();
  
  // Get GenParticles
  edm::Handle<edm::View<reco::GenParticle> > genParticles;
  iEvent.getByToken(genToken_, genParticles);
  
  // Get stuff for conversions
  edm::Handle<reco::ConversionCollection> convs;
  edm::Handle<reco::BeamSpot> thebs;
  iEvent.getByToken(convToken_, convs);
  iEvent.getByToken(beamToken_, thebs);
  
  for( size_t i = 0; i < collection->size(); ++i ) {
    auto el = collection->refAt(i);
    pt_ = el->pt();
    etaSC_ = el->superCluster()->eta();
    
    // ID and matching
    dEtaIn_ = el->deltaEtaSuperClusterTrackAtVtx();
    dPhiIn_ = el->deltaPhiSuperClusterTrackAtVtx();
    hOverE_ = el->hcalOverEcal();
    sigmaIetaIeta_ = el->sigmaIetaIeta();
    full5x5_sigmaIetaIeta_ = el->full5x5_sigmaIetaIeta();
    // |1/E-1/p| = |1/E - EoverPinner/E| is computed below
    // The if protects against ecalEnergy == inf or zero (always
    // the case for electrons below 5 GeV in miniAOD)
    if( el->ecalEnergy() == 0 ){
      printf("Electron energy is zero!\n");
      ooEmooP_ = 1e30;
    }else if( !std::isfinite(el->ecalEnergy())){
      printf("Electron energy is not finite!\n");
      ooEmooP_ = 1e30;
    }else{
      ooEmooP_ = fabs(1.0/el->ecalEnergy() - el->eSuperClusterOverP()/el->ecalEnergy() );
    }
    
    // Isolation
    reco::GsfElectron::PflowIsolationVariables pfIso = el->pfIsolationVariables();
    // Compute isolation with delta beta correction for PU
    float absiso = pfIso.sumChargedHadronPt 
      + std::max(0.0 , pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt - 0.5 * pfIso.sumPUPt );
    relIsoWithDBeta_ = absiso/pt_;
    
    // Impact parameter
    d0_ = (-1) * el->gsfTrack()->dxy(pv.position() );
    dz_ = el->gsfTrack()->dz( pv.position() );
    
    // Conversion rejection
    constexpr reco::HitPattern::HitCategory missingHitType =
      reco::HitPattern::MISSING_INNER_HITS;
    expectedMissingInnerHits_ = el->gsfTrack()->hitPattern().numberOfHits(missingHitType);
    passConversionVeto_ = false;
    if( thebs.isValid() && convs.isValid() ) {
      passConversionVeto_ = !ConversionTools::hasMatchedConversion(*el,convs,
								   thebs->position());
    }else{
      printf("\n\nERROR!!! conversions not found!!!\n");
    }
    
    isTrue_ = matchToTruth(*el, genParticles);
    isPass_ = ( el->electronID(electronIdTag_.encode()) > 0.5 );
    
    electronTree_->Fill();
  }  
}


// ------------ method called once each job just before starting event loop  ------------
void 
MiniAODElectronIDValidationAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MiniAODElectronIDValidationAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
void 
MiniAODElectronIDValidationAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
MiniAODElectronIDValidationAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
MiniAODElectronIDValidationAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
MiniAODElectronIDValidationAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MiniAODElectronIDValidationAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// The function that uses algorith from Josh Bendavid with 
// an explicit loop over gen particles. 
int MiniAODElectronIDValidationAnalyzer::matchToTruth(const reco::GsfElectron &el, 
				    const edm::Handle<edm::View<reco::GenParticle>> &genParticles){

  // 
  // Explicit loop and geometric matching method (advised by Josh Bendavid)
  //

  // Find the closest status 1 gen electron to the reco electron
  double dR = 999;
  const reco::Candidate *closestElectron = 0;
  for(size_t i=0; i<genParticles->size();i++){
    const reco::Candidate *particle = &(*genParticles)[i];
    // Drop everything that is not electron or not status 1
    if( abs(particle->pdgId()) != 11 || particle->status() != 1 )
      continue;
    //
    double dRtmp = ROOT::Math::VectorUtil::DeltaR( el.p4(), particle->p4() );
    if( dRtmp < dR ){
      dR = dRtmp;
      closestElectron = particle;
    }
  }
  // See if the closest electron (if it exists) is close enough.
  // If not, no match found.
  if( !(closestElectron != 0 && dR < 0.1) ) {
    return UNMATCHED;
  }

  // 
  int ancestorPID = -999; 
  int ancestorStatus = -999;
  findFirstNonElectronMother(closestElectron, ancestorPID, ancestorStatus);

  if( ancestorPID == -999 && ancestorStatus == -999 ){
    // No non-electron parent??? This should never happen.
    // Complain.
    printf("ElectronNtupler: ERROR! Electron does not apper to have a non-electron parent\n");
    return UNMATCHED;
  }
  
  if( abs(ancestorPID) > 50 && ancestorStatus == 2 )
    return TRUE_NON_PROMPT_ELECTRON;

  if( abs(ancestorPID) == 15 && ancestorStatus == 2 )
    return TRUE_ELECTRON_FROM_TAU;

  // What remains is true prompt electrons
  return TRUE_PROMPT_ELECTRON;
}

void MiniAODElectronIDValidationAnalyzer::findFirstNonElectronMother(const reco::Candidate *particle,
						   int &ancestorPID, int &ancestorStatus){
  
  if( particle == 0 ){
    printf("ElectronNtupler: ERROR! null candidate pointer, this should never happen\n");
    return;
  }
  
  // Is this the first non-electron parent? If yes, return, otherwise
  // go deeper into recursion
  if( abs(particle->pdgId()) == 11 ){
    findFirstNonElectronMother(particle->mother(0), ancestorPID, ancestorStatus);
  }else{
    ancestorPID = particle->pdgId();
    ancestorStatus = particle->status();
  }

  return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MiniAODElectronIDValidationAnalyzer);
