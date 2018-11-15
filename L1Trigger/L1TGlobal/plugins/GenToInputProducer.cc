////
/// \class l1t::GenToInputProducer
///
/// Description: Create Input Collections for the GT from MC gen particles.  Allows testing of emulation.
///
/// 
/// \author: D. Puigh OSU
///
///  Modeled after FakeInputProducer.cc


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include <vector>
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"

#include "TMath.h"
#include "TRandom3.h"
#include <cstdlib>

using namespace std;
using namespace edm;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace l1t {

//
// class declaration
//

  class GenToInputProducer : public EDProducer {
  public:
    explicit GenToInputProducer(const ParameterSet&);
    ~GenToInputProducer() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void produce(Event&, EventSetup const&) override;
    void beginJob() override;
    void endJob() override;
    void beginRun(Run const&iR, EventSetup const&iE) override;
    void endRun(Run const& iR, EventSetup const& iE) override;

    int convertPhiToHW(double iphi, int steps);
    int convertEtaToHW(double ieta, double minEta, double maxEta, int steps);
    int convertPtToHW(double ipt, int maxPt, double step);

    // ----------member data ---------------------------
    unsigned long long m_paramsCacheId; // Cache-ID from current parameters, to check if needs to be updated.
    //std::shared_ptr<const CaloParams> m_dbpars; // Database parameters for the trigger, to be updated as needed.
    //std::shared_ptr<const FirmwareVersion> m_fwv;
    //std::shared_ptr<FirmwareVersion> m_fwv; //not const during testing.

    TRandom3* gRandom;

    // BX parameters
    int bxFirst_;
    int bxLast_;

    int maxNumMuCands_;  
    int maxNumJetCands_; 
    int maxNumEGCands_;  
    int maxNumTauCands_;
     
    double jetEtThreshold_;
    double tauEtThreshold_;
    double egEtThreshold_;
    double muEtThreshold_;

    // Control how to end the job 
    int emptyBxTrailer_;
    int emptyBxEvt_;
    int eventCnt_;

    // Tokens
    edm::EDGetTokenT <reco::GenParticleCollection> genParticlesToken;
    edm::EDGetTokenT <reco::GenJetCollection> genJetsToken;
    edm::EDGetTokenT <reco::GenMETCollection> genMetToken;

    int counter_;

    std::vector<l1t::Muon> muonVec_bxm2;
    std::vector<l1t::Muon> muonVec_bxm1;
    std::vector<l1t::Muon> muonVec_bx0;
    std::vector<l1t::Muon> muonVec_bxp1;

    std::vector<l1t::EGamma> egammaVec_bxm2;
    std::vector<l1t::EGamma> egammaVec_bxm1;
    std::vector<l1t::EGamma> egammaVec_bx0;
    std::vector<l1t::EGamma> egammaVec_bxp1;

    std::vector<l1t::Tau> tauVec_bxm2;
    std::vector<l1t::Tau> tauVec_bxm1;
    std::vector<l1t::Tau> tauVec_bx0;
    std::vector<l1t::Tau> tauVec_bxp1;

    std::vector<l1t::Jet> jetVec_bxm2;
    std::vector<l1t::Jet> jetVec_bxm1;
    std::vector<l1t::Jet> jetVec_bx0;
    std::vector<l1t::Jet> jetVec_bxp1;

    std::vector<l1t::EtSum> etsumVec_bxm2;
    std::vector<l1t::EtSum> etsumVec_bxm1;
    std::vector<l1t::EtSum> etsumVec_bx0;
    std::vector<l1t::EtSum> etsumVec_bxp1;
    
    GlobalExtBlk extCond_bxm2;
    GlobalExtBlk extCond_bxm1;
    GlobalExtBlk extCond_bx0;
    GlobalExtBlk extCond_bxp1;    

  };

  //
  // constructors and destructor
  //
  GenToInputProducer::GenToInputProducer(const ParameterSet& iConfig)
  {
    // register what you produce
    produces<BXVector<l1t::EGamma>>();
    produces<BXVector<l1t::Muon>>();
    produces<BXVector<l1t::Tau>>();
    produces<BXVector<l1t::Jet>>();
    produces<BXVector<l1t::EtSum>>();
    produces<GlobalExtBlkBxCollection>();

    // Setup parameters
    bxFirst_ = iConfig.getParameter<int>("bxFirst");
    bxLast_  = iConfig.getParameter<int>("bxLast");

    maxNumMuCands_  = iConfig.getParameter<int>("maxMuCand");
    maxNumJetCands_ = iConfig.getParameter<int>("maxJetCand");
    maxNumEGCands_  = iConfig.getParameter<int>("maxEGCand");
    maxNumTauCands_ = iConfig.getParameter<int>("maxTauCand");

    jetEtThreshold_ = iConfig.getParameter<double>("jetEtThreshold");
    tauEtThreshold_ = iConfig.getParameter<double>("tauEtThreshold");
    egEtThreshold_  = iConfig.getParameter<double>("egEtThreshold");
    muEtThreshold_  = iConfig.getParameter<double>("muEtThreshold");


    emptyBxTrailer_   = iConfig.getParameter<int>("emptyBxTrailer");
    emptyBxEvt_       = iConfig.getParameter<int>("emptyBxEvt");


    genParticlesToken = consumes <reco::GenParticleCollection> (std::string("genParticles"));
    genJetsToken      = consumes <reco::GenJetCollection> (std::string("ak4GenJets"));
    genMetToken       = consumes <reco::GenMETCollection> (std::string("genMetCalo"));   


    // set cache id to zero, will be set at first beginRun:
    m_paramsCacheId = 0;
    eventCnt_ = 0;
  }


  GenToInputProducer::~GenToInputProducer()
  {
  }



//
// member functions
//

// ------------ method called to produce the data ------------
void
GenToInputProducer::produce(Event& iEvent, const EventSetup& iSetup)
{

  eventCnt_++;

  LogDebug("GtGenToInputProducer") << "GenToInputProducer::produce function called...\n";

  // Setup vectors
  std::vector<l1t::Muon> muonVec;
  std::vector<l1t::EGamma> egammaVec;
  std::vector<l1t::Tau> tauVec;
  std::vector<l1t::Jet> jetVec;
  std::vector<l1t::EtSum> etsumVec;
  GlobalExtBlk extCond_bx;

  // Set the range of BX....TO DO...move to Params or determine from param set.
  int bxFirst = bxFirst_;
  int bxLast  = bxLast_;


  // Default values objects
  double MaxLepPt_ = 255;
  double MaxJetPt_ = 1023;
  double MaxEt_ = 2047;

  double MaxCaloEta_ = 5.0;
  double MaxMuonEta_ = 2.45;

  double PhiStepCalo_ = 144;
  double PhiStepMuon_ = 576;

  // eta scale
  double EtaStepCalo_ = 230;
  double EtaStepMuon_ = 450;

  // Et scale (in GeV)
  double PtStep_ = 0.5;


  //outputs
  std::unique_ptr<l1t::EGammaBxCollection> egammas (new l1t::EGammaBxCollection(0, bxFirst, bxLast));
  std::unique_ptr<l1t::MuonBxCollection> muons (new l1t::MuonBxCollection(0, bxFirst, bxLast));
  std::unique_ptr<l1t::TauBxCollection> taus (new l1t::TauBxCollection(0, bxFirst, bxLast));
  std::unique_ptr<l1t::JetBxCollection> jets (new l1t::JetBxCollection(0, bxFirst, bxLast));
  std::unique_ptr<l1t::EtSumBxCollection> etsums (new l1t::EtSumBxCollection(0, bxFirst, bxLast));
  std::unique_ptr<GlobalExtBlkBxCollection> extCond( new GlobalExtBlkBxCollection(0,bxFirst,bxLast));

  std::vector<int> mu_cands_index;
  std::vector<int> eg_cands_index;
  std::vector<int> tau_cands_index;
  edm::Handle<reco::GenParticleCollection > genParticles;
  // Make sure that you can get genParticles
  if( iEvent.getByToken(genParticlesToken, genParticles) ){

    for( size_t k = 0; k < genParticles->size(); k++ ){
      const reco::Candidate & mcParticle = (*genParticles)[k];

      int status = mcParticle.status();
      int pdgId  = mcParticle.pdgId();
      double pt  = mcParticle.pt();
      
      // Only use status 1 particles  (Tau's need to be allowed through..take status 2 taus)
      if( status!=1 && !(abs(pdgId)==15 && status==2) ) continue;

      int absId = abs(pdgId);

      if( absId==11 && pt>=egEtThreshold_ )       eg_cands_index.push_back(k);
      else if( absId==13 && pt>=muEtThreshold_ )  mu_cands_index.push_back(k);
      else if( absId==15 && pt>=tauEtThreshold_ ) tau_cands_index.push_back(k);
    }
  }
  else {
    LogTrace("GtGenToInputProducer") << ">>> GenParticles collection not found!" << std::endl;
  }



  // Muon Collection
  int numMuCands = int( mu_cands_index.size() );
  Int_t idxMu[numMuCands];
  double muPtSorted[numMuCands];
  for( int iMu=0; iMu<numMuCands; iMu++ ) muPtSorted[iMu] = genParticles->at(mu_cands_index[iMu]).pt();

  TMath::Sort(numMuCands,muPtSorted,idxMu);
  for( int iMu=0; iMu<numMuCands; iMu++ ){

    if( iMu>=maxNumMuCands_ ) continue;
  
    const reco::Candidate & mcParticle = (*genParticles)[mu_cands_index[idxMu[iMu]]];

    int pt   = convertPtToHW( mcParticle.pt(), MaxLepPt_, PtStep_ );
    int eta  = convertEtaToHW( mcParticle.eta(), -MaxMuonEta_, MaxMuonEta_, EtaStepMuon_);
    int phi  = convertPhiToHW( mcParticle.phi(), PhiStepMuon_ );
    int qual = gRandom->Integer(16);//4;
    int iso  = gRandom->Integer(4)%2;//1;
    int charge = ( mcParticle.charge()<0 ) ? 1 : 0;
    int chargeValid = 1;
    int tfMuIdx = 0;
    int tag = 1;
    bool debug = false;
    int isoSum = 0;
    int dPhi = 0;
    int dEta = 0;
    int rank = 0;
    int hwEtaAtVtx = eta;
    int hwPhiAtVtx = phi;

    // Eta outside of acceptance
    if( eta>=9999 ) continue;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

    l1t::Muon mu(*p4, pt, eta, phi, qual, charge, chargeValid, iso, tfMuIdx, tag, debug, isoSum, dPhi, dEta, rank, hwEtaAtVtx, hwPhiAtVtx);
    muonVec.push_back(mu);
  }


  // EG Collection
  int numEgCands = int( eg_cands_index.size() );
  Int_t idxEg[numEgCands];
  double egPtSorted[numEgCands];
  for( int iEg=0; iEg<numEgCands; iEg++ ) egPtSorted[iEg] = genParticles->at(eg_cands_index[iEg]).pt();

  TMath::Sort(numEgCands,egPtSorted,idxEg);
  for( int iEg=0; iEg<numEgCands; iEg++ ){

    if( iEg>=maxNumEGCands_ ) continue;
  
    const reco::Candidate & mcParticle = (*genParticles)[eg_cands_index[idxEg[iEg]]];

    int pt   = convertPtToHW( mcParticle.pt(), MaxLepPt_, PtStep_ );
    int eta  = convertEtaToHW( mcParticle.eta(), -MaxCaloEta_, MaxCaloEta_, EtaStepCalo_ );
    int phi  = convertPhiToHW( mcParticle.phi(), PhiStepCalo_ );
    int qual = 1;
    int iso  = gRandom->Integer(4)%2;

    // Eta outside of acceptance
    if( eta>=9999 ) continue;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

    l1t::EGamma eg(*p4, pt, eta, phi, qual, iso);
    egammaVec.push_back(eg);
  }
  


  // Tau Collection
  int numTauCands = int( tau_cands_index.size() );
  Int_t idxTau[numTauCands];
  double tauPtSorted[numTauCands];
  for( int iTau=0; iTau<numTauCands; iTau++ ) tauPtSorted[iTau] = genParticles->at(tau_cands_index[iTau]).pt();

  TMath::Sort(numTauCands,tauPtSorted,idxTau);
  for( int iTau=0; iTau<numTauCands; iTau++ ){

    if( iTau>=maxNumTauCands_ ) continue;
  
    const reco::Candidate & mcParticle = (*genParticles)[tau_cands_index[idxTau[iTau]]];

    int pt   = convertPtToHW( mcParticle.pt(), MaxLepPt_, PtStep_ );
    int eta  = convertEtaToHW( mcParticle.eta(), -MaxCaloEta_, MaxCaloEta_, EtaStepCalo_);
    int phi  = convertPhiToHW( mcParticle.phi(), PhiStepCalo_ );
    int qual = 1;
    int iso  = gRandom->Integer(4)%2;

    // Eta outside of acceptance
    if( eta>=9999 ) continue;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

    l1t::Tau tau(*p4, pt, eta, phi, qual, iso);
    tauVec.push_back(tau);
  }


  // Temporary hack to increase number of EGs and taus
  int maxOtherEGs = 4;
  int maxOtherTaus = 8;
  int numCurrentEGs  = int( egammaVec.size() );
  int numCurrentTaus = int( tauVec.size() );

  int numExtraEGs=0, numExtraTaus=0;
  // end hack

  // Use to sum the energy of the objects in the event for ETT and HTT
  // sum all jets
  double sumEt = 0;

  int nJet = 0;
  edm::Handle<reco::GenJetCollection > genJets;
  // Make sure that you can get genJets
  if( iEvent.getByToken(genJetsToken, genJets) ){ // Jet Collection
    for(reco::GenJetCollection::const_iterator genJet = genJets->begin(); genJet!=genJets->end(); ++genJet ){

      //Keep running sum of total Et
      sumEt += genJet->et(); 

      // Apply pt and eta cut?
      if( genJet->pt()<jetEtThreshold_ ) continue;

      //
      if( nJet>=maxNumJetCands_ ) continue;
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

      int pt  = convertPtToHW( genJet->et(), MaxJetPt_, PtStep_ );
      int eta = convertEtaToHW( genJet->eta(), -MaxCaloEta_, MaxCaloEta_, EtaStepCalo_ );
      int phi = convertPhiToHW( genJet->phi(), PhiStepCalo_ );

      // Eta outside of acceptance
      if( eta>=9999 ) continue;

      int qual = 0;

      l1t::Jet jet(*p4, pt, eta, phi, qual);
      jetVec.push_back(jet);

      nJet++;

      // Temporary hack to increase number of EGs and taus
      if( (numExtraEGs+numCurrentEGs)<maxNumEGCands_ && numExtraEGs<maxOtherEGs ){
	numExtraEGs++;

	int EGpt   = convertPtToHW( genJet->et(), MaxLepPt_, PtStep_ );
	int EGeta  = convertEtaToHW( genJet->eta(), -MaxCaloEta_, MaxCaloEta_, EtaStepCalo_ );
	int EGphi  = convertPhiToHW( genJet->phi(), PhiStepCalo_ );

	int EGqual = 1;
	int EGiso  = gRandom->Integer(4)%2;

	l1t::EGamma eg(*p4, EGpt, EGeta, EGphi, EGqual, EGiso);
	egammaVec.push_back(eg);
      }

      if( (numExtraTaus+numCurrentTaus)<maxNumTauCands_ && numExtraTaus<maxOtherTaus ){
	numExtraTaus++;

	int Taupt   = convertPtToHW( genJet->et(), MaxLepPt_, PtStep_ );
	int Taueta  = convertEtaToHW( genJet->eta(), -MaxCaloEta_, MaxCaloEta_, EtaStepCalo_ );
	int Tauphi  = convertPhiToHW( genJet->phi(), PhiStepCalo_ );
	int Tauqual = 1;
	int Tauiso  = gRandom->Integer(4)%2;

	l1t::Tau tau(*p4, Taupt, Taueta, Tauphi, Tauqual, Tauiso);
	tauVec.push_back(tau);
      }
      // end hack
    }
  }
  else {
    LogTrace("GtGenToInputProducer") << ">>> GenJets collection not found!" << std::endl;
  }


// Put the total Et into EtSums  (Make HTT slightly smaller to tell them apart....not supposed to be realistic) 
   int pt  = convertPtToHW( sumEt, 2047, PtStep_ );
   ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
   l1t::EtSum etTotal(*p4, l1t::EtSum::EtSumType::kTotalEt,pt, 0, 0, 0);    

// Scale down ETTem as an estimate
   pt  = convertPtToHW( sumEt*0.6, 2047, PtStep_ );
   l1t::EtSum etEmTotal(*p4, l1t::EtSum::EtSumType::kTotalEtEm,pt, 0, 0, 0);

   //ccla Generate uniform distribution of tower counts
   int nTowers=4095*gRandom->Rndm();
   l1t::EtSum towerCounts(*p4, l1t::EtSum::EtSumType::kTowerCount,nTowers, 0, 0, 0);

   //ccla Generate uniform distributions of AsymEt, AsymHt, AsymEtHF, AsymHtF
   int nAsymEt=255*gRandom->Rndm();
   l1t::EtSum AsymEt(*p4, l1t::EtSum::EtSumType::kAsymEt,nAsymEt, 0, 0, 0);
   int nAsymHt=255*gRandom->Rndm();
   l1t::EtSum AsymHt(*p4, l1t::EtSum::EtSumType::kAsymHt,nAsymHt, 0, 0, 0);
   int nAsymEtHF=255*gRandom->Rndm();
   l1t::EtSum AsymEtHF(*p4, l1t::EtSum::EtSumType::kAsymEtHF,nAsymEtHF, 0, 0, 0);
   int nAsymHtHF=255*gRandom->Rndm();
   l1t::EtSum AsymHtHF(*p4, l1t::EtSum::EtSumType::kAsymHtHF,nAsymHtHF, 0, 0, 0);

   pt  = convertPtToHW( sumEt*0.9, 2047, PtStep_ );
   l1t::EtSum htTotal(*p4, l1t::EtSum::EtSumType::kTotalHt,pt, 0, 0, 0); 

// Add EtSums for testing the MinBias Trigger (use some random numbers)   
   int hfP0val  = gRandom->Poisson(4.);
   if(hfP0val>15) hfP0val = 15;
   l1t::EtSum hfP0(*p4, l1t::EtSum::EtSumType::kMinBiasHFP0,hfP0val, 0, 0, 0); 

   int hfM0val  = gRandom->Poisson(4.);
   if(hfM0val>15) hfM0val = 15;
   l1t::EtSum hfM0(*p4, l1t::EtSum::EtSumType::kMinBiasHFM0,hfM0val, 0, 0, 0); 

   int hfP1val  = gRandom->Poisson(4.);
   if(hfP1val>15) hfP1val = 15;
   l1t::EtSum hfP1(*p4, l1t::EtSum::EtSumType::kMinBiasHFP1,hfP1val, 0, 0, 0); 

   int hfM1val  = gRandom->Poisson(4.);
   if(hfM1val>15) hfM1val = 15;
   l1t::EtSum hfM1(*p4, l1t::EtSum::EtSumType::kMinBiasHFM1,hfM1val, 0, 0, 0); 

// Do same for Centrality
   int cent30val(0), cent74val(0);
   int centa  = gRandom->Poisson(2.);
   int centb  = gRandom->Poisson(2.);
   if (centa >= centb) {
     cent30val=centa;
     cent74val=centb;
   }else{
     cent30val=centb;
     cent74val=centa;
   }

   if(cent30val>15) cent30val = 15;
   if(cent74val>15) cent74val = 15;

   int shift = 4;
   int centralval=0;
   centralval |= cent30val & 0xF;
   centralval |= (cent74val & 0xF ) << shift;

   l1t::EtSum centrality(*p4, l1t::EtSum::EtSumType::kCentrality,centralval, 0, 0, 0); 

   int mpt = 0;
   int mphi= 0;
   int mptHf = 0;
   int mphiHf= 0;   
   int mhpt = 0;
   int mhphi= 0;  
   int mhptHf = 0;
   int mhphiHf= 0;  

   edm::Handle<reco::GenMETCollection> genMet;
   // Make sure that you can get genMET
   if( iEvent.getByToken(genMetToken, genMet) ){
     mpt  = convertPtToHW( genMet->front().pt(), MaxEt_, PtStep_ );
     mphi = convertPhiToHW( genMet->front().phi(), PhiStepCalo_ );

     // Make Missing Et with HF slightly largeer and rotated (These are all fake inputs anyway...not supposed to be realistic)
     mptHf  = convertPtToHW( genMet->front().pt()*1.1, MaxEt_, PtStep_ );
     mphiHf = convertPhiToHW( genMet->front().phi()+ 3.14/7., PhiStepCalo_ );

     // Make Missing Ht slightly smaller and rotated (These are all fake inputs anyway...not supposed to be realistic)
     mhpt  = convertPtToHW( genMet->front().pt()*0.9, MaxEt_, PtStep_ );
     mhphi = convertPhiToHW( genMet->front().phi()+ 3.14/5., PhiStepCalo_ );

     // Ditto with Hissing Ht with HF
     mhptHf  = convertPtToHW( genMet->front().pt()*0.95, MaxEt_, PtStep_ );
     mhphiHf = convertPhiToHW( genMet->front().phi()+ 3.14/6., PhiStepCalo_ );
   }
   else {
     LogTrace("GtGenToInputProducer") << ">>> GenMet collection not found!" << std::endl;
   }

// Missing Et and missing htt
   l1t::EtSum etmiss(*p4, l1t::EtSum::EtSumType::kMissingEt,mpt, 0,mphi, 0); 
   l1t::EtSum etmissHF(*p4, l1t::EtSum::EtSumType::kMissingEtHF,mptHf, 0,mphiHf, 0);    
   l1t::EtSum htmiss(*p4, l1t::EtSum::EtSumType::kMissingHt,mhpt, 0,mhphi, 0); 
   l1t::EtSum htmissHF(*p4, l1t::EtSum::EtSumType::kMissingHtHF,mhptHf, 0,mhphiHf, 0); 

// Fill the EtSums in the Correct order
   etsumVec.push_back(etTotal); 
   etsumVec.push_back(etEmTotal);
   etsumVec.push_back(hfP0); // Frame0

   etsumVec.push_back(htTotal);
   etsumVec.push_back(towerCounts);
   etsumVec.push_back(hfM0); //Frame1

   etsumVec.push_back(etmiss);
   etsumVec.push_back(AsymEt);
   etsumVec.push_back(hfP1); //Frame2

   etsumVec.push_back(htmiss);
   etsumVec.push_back(AsymHt);
   etsumVec.push_back(hfM1); //Frame3

   etsumVec.push_back(etmissHF);
   etsumVec.push_back(AsymEtHF); // Frame4

   etsumVec.push_back(htmissHF); 
   etsumVec.push_back(AsymHtHF);
   etsumVec.push_back(centrality); // Frame5
 
// Fill in some external conditions for testing
   if((iEvent.id().event())%2 == 0 ) {
     for(int i=0; i<255; i=i+2) extCond_bx.setExternalDecision(i,true); 
   } else {
     for(int i=1; i<255; i=i+2) extCond_bx.setExternalDecision(i,true);
   }
 
   // Insert all the bx into the L1 Collections
   //printf("Event %i  EmptyBxEvt %i emptyBxTrailer %i diff %i \n",eventCnt_,emptyBxEvt_,emptyBxTrailer_,(emptyBxEvt_ - eventCnt_));

   // Fill Muons
   for( int iMu=0; iMu<int(muonVec_bxm2.size()); iMu++ ){
     muons->push_back(-2, muonVec_bxm2[iMu]);
   }
   for( int iMu=0; iMu<int(muonVec_bxm1.size()); iMu++ ){
     muons->push_back(-1, muonVec_bxm1[iMu]);
   }
   for( int iMu=0; iMu<int(muonVec_bx0.size()); iMu++ ){
     muons->push_back(0, muonVec_bx0[iMu]);
   }
   for( int iMu=0; iMu<int(muonVec_bxp1.size()); iMu++ ){
     muons->push_back(1, muonVec_bxp1[iMu]);
   }
   if(emptyBxTrailer_<=(emptyBxEvt_ - eventCnt_)) {
     for( int iMu=0; iMu<int(muonVec.size()); iMu++ ){
        muons->push_back(2, muonVec[iMu]);
     }
   } else {
     // this event is part of empty trailer...clear out data
     muonVec.clear();  
   }  

   // Fill Egammas
   for( int iEG=0; iEG<int(egammaVec_bxm2.size()); iEG++ ){
     egammas->push_back(-2, egammaVec_bxm2[iEG]);
   }
   for( int iEG=0; iEG<int(egammaVec_bxm1.size()); iEG++ ){
     egammas->push_back(-1, egammaVec_bxm1[iEG]);
   }
   for( int iEG=0; iEG<int(egammaVec_bx0.size()); iEG++ ){
     egammas->push_back(0, egammaVec_bx0[iEG]);
   }
   for( int iEG=0; iEG<int(egammaVec_bxp1.size()); iEG++ ){
     egammas->push_back(1, egammaVec_bxp1[iEG]);
   }
   if(emptyBxTrailer_<=(emptyBxEvt_ - eventCnt_)) {
     for( int iEG=0; iEG<int(egammaVec.size()); iEG++ ){
        egammas->push_back(2, egammaVec[iEG]);
     }
   } else {
     // this event is part of empty trailer...clear out data
     egammaVec.clear();  
   }  

   // Fill Taus
   for( int iTau=0; iTau<int(tauVec_bxm2.size()); iTau++ ){
     taus->push_back(-2, tauVec_bxm2[iTau]);
   }
   for( int iTau=0; iTau<int(tauVec_bxm1.size()); iTau++ ){
     taus->push_back(-1, tauVec_bxm1[iTau]);
   }
   for( int iTau=0; iTau<int(tauVec_bx0.size()); iTau++ ){
     taus->push_back(0, tauVec_bx0[iTau]);
   }
   for( int iTau=0; iTau<int(tauVec_bxp1.size()); iTau++ ){
     taus->push_back(1, tauVec_bxp1[iTau]);
   }
   if(emptyBxTrailer_<=(emptyBxEvt_ - eventCnt_)) {
     for( int iTau=0; iTau<int(tauVec.size()); iTau++ ){
        taus->push_back(2, tauVec[iTau]);
     }
   } else {
     // this event is part of empty trailer...clear out data
     tauVec.clear();  
   }  

   // Fill Jets
   for( int iJet=0; iJet<int(jetVec_bxm2.size()); iJet++ ){
     jets->push_back(-2, jetVec_bxm2[iJet]);
   }
   for( int iJet=0; iJet<int(jetVec_bxm1.size()); iJet++ ){
     jets->push_back(-1, jetVec_bxm1[iJet]);
   }
   for( int iJet=0; iJet<int(jetVec_bx0.size()); iJet++ ){
     jets->push_back(0, jetVec_bx0[iJet]);
   }
   for( int iJet=0; iJet<int(jetVec_bxp1.size()); iJet++ ){
     jets->push_back(1, jetVec_bxp1[iJet]);
   }
   if(emptyBxTrailer_<=(emptyBxEvt_ - eventCnt_)) {
     for( int iJet=0; iJet<int(jetVec.size()); iJet++ ){
        jets->push_back(2, jetVec[iJet]);
     }
   } else {
     // this event is part of empty trailer...clear out data
     jetVec.clear();  
   }  

   // Fill Etsums
   for( int iETsum=0; iETsum<int(etsumVec_bxm2.size()); iETsum++ ){
     etsums->push_back(-2, etsumVec_bxm2[iETsum]);
   }
   for( int iETsum=0; iETsum<int(etsumVec_bxm1.size()); iETsum++ ){
     etsums->push_back(-1, etsumVec_bxm1[iETsum]);
   }
   for( int iETsum=0; iETsum<int(etsumVec_bx0.size()); iETsum++ ){
     etsums->push_back(0, etsumVec_bx0[iETsum]);
   }
   for( int iETsum=0; iETsum<int(etsumVec_bxp1.size()); iETsum++ ){
     etsums->push_back(1, etsumVec_bxp1[iETsum]);
   }
   if(emptyBxTrailer_<=(emptyBxEvt_ - eventCnt_)) {
     for( int iETsum=0; iETsum<int(etsumVec.size()); iETsum++ ){
        etsums->push_back(2, etsumVec[iETsum]);
     }
   } else {
     // this event is part of empty trailer...clear out data
     etsumVec.clear();  
   }  

   // Fill Externals
   extCond->push_back(-2, extCond_bxm2);
   extCond->push_back(-1, extCond_bxm1);
   extCond->push_back(0,  extCond_bx0);
   extCond->push_back(1,  extCond_bxp1);
   if(emptyBxTrailer_<=(emptyBxEvt_ - eventCnt_)) {
     extCond->push_back(2,  extCond_bx);
   } else {
      // this event is part of the empty trailer...clear out data
      extCond_bx.reset();
   }     
   

  iEvent.put(std::move(egammas));
  iEvent.put(std::move(muons));
  iEvent.put(std::move(taus));
  iEvent.put(std::move(jets));
  iEvent.put(std::move(etsums));
  iEvent.put(std::move(extCond));

  // Now shift the bx data by one to prepare for next event.
  muonVec_bxm2 = muonVec_bxm1;
  egammaVec_bxm2 = egammaVec_bxm1;
  tauVec_bxm2 = tauVec_bxm1;
  jetVec_bxm2 = jetVec_bxm1;
  etsumVec_bxm2 = etsumVec_bxm1;
  extCond_bxm2 = extCond_bxm1;

  muonVec_bxm1 = muonVec_bx0;
  egammaVec_bxm1 = egammaVec_bx0;
  tauVec_bxm1 = tauVec_bx0;
  jetVec_bxm1 = jetVec_bx0;
  etsumVec_bxm1 = etsumVec_bx0;
  extCond_bxm1 = extCond_bx0;

  muonVec_bx0 = muonVec_bxp1;
  egammaVec_bx0 = egammaVec_bxp1;
  tauVec_bx0 = tauVec_bxp1;
  jetVec_bx0 = jetVec_bxp1;
  etsumVec_bx0 = etsumVec_bxp1;
  extCond_bx0 = extCond_bxp1;

  muonVec_bxp1 = muonVec;
  egammaVec_bxp1 = egammaVec;
  tauVec_bxp1 = tauVec;
  jetVec_bxp1 = jetVec;
  etsumVec_bxp1 = etsumVec;
  extCond_bxp1 = extCond_bx;
}

// ------------ method called once each job just before starting event loop ------------
void
GenToInputProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop ------------
void
GenToInputProducer::endJob() {
}

// ------------ method called when starting to processes a run ------------

void GenToInputProducer::beginRun(Run const&iR, EventSetup const&iE){

  LogDebug("GtGenToInputProducer") << "GenToInputProducer::beginRun function called...\n";

  counter_ = 0;
  srand( 0 );

  gRandom = new TRandom3();
}

// ------------ method called when ending the processing of a run ------------
void GenToInputProducer::endRun(Run const& iR, EventSetup const& iE){

}


// ------------ methods to convert from physical to HW values ------------
int GenToInputProducer::convertPhiToHW(double iphi, int steps){

  double phiMax = 2 * M_PI;
  if( iphi < 0 ) iphi += 2*M_PI;
  if( iphi > phiMax) iphi -= phiMax;

  int hwPhi = int( (iphi/phiMax)*steps + 0.00001 );
  return hwPhi;
}

int GenToInputProducer::convertEtaToHW(double ieta, double minEta, double maxEta, int steps){

   double binWidth = (maxEta - minEta)/steps;
     
   //if we are outside the limits, set error
   if(ieta < minEta) return 99999;//ieta = minEta+binWidth/2.;
   if(ieta > maxEta) return 99999;//ieta = maxEta-binWidth/2.;
      
   int binNum = (int)(ieta/binWidth);
   if(ieta<0.) binNum--;
      
//   unsigned int hwEta = binNum & bitMask; 
//   Remove masking for BXVectors...only assume in raw data

  return binNum;
}

int GenToInputProducer::convertPtToHW(double ipt, int maxPt, double step){

  int hwPt = int( ipt/step + 0.0001 );
  // if above max Pt, set to largest value
  if( hwPt > maxPt ) hwPt = maxPt;

  return hwPt;
}


// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void
GenToInputProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

} // namespace

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::GenToInputProducer);
