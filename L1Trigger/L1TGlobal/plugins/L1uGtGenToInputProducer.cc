///
/// \class l1t::L1uGtGenToInputProducer
///
/// Description: Create Input Collections for the GT from MC gen particles.  Allows testing of emulation.
///
/// 
/// \author: D. Puigh OSU
///
///  Modeled after L1TGlobalFakeInputProducer.cc


// system include files
#include <boost/shared_ptr.hpp>

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

#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"

#include "TMath.h"

using namespace std;
using namespace edm;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace l1t {

//
// class declaration
//

  class L1uGtGenToInputProducer : public EDProducer {
  public:
    explicit L1uGtGenToInputProducer(const ParameterSet&);
    ~L1uGtGenToInputProducer();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual void produce(Event&, EventSetup const&);
    virtual void beginJob();
    virtual void endJob();
    virtual void beginRun(Run const&iR, EventSetup const&iE);
    virtual void endRun(Run const& iR, EventSetup const& iE);

    int convertPhiToHW(double iphi, int steps);
    unsigned int convertEtaToHW(double ieta, double minEta, double maxEta, int steps, unsigned int bitMask);
    int convertPtToHW(double ipt, double maxPt, int steps);

    // ----------member data ---------------------------
    unsigned long long m_paramsCacheId; // Cache-ID from current parameters, to check if needs to be updated.
    //boost::shared_ptr<const CaloParams> m_dbpars; // Database parameters for the trigger, to be updated as needed.
    //boost::shared_ptr<const FirmwareVersion> m_fwv;
    //boost::shared_ptr<FirmwareVersion> m_fwv; //not const during testing.

    // BX parameters
    int bxFirst_;
    int bxLast_;

    double jetEtThreshold_;
    double tauEtThreshold_;
    double egEtThreshold_;
    double muEtThreshold_;

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

  };

  //
  // constructors and destructor
  //
  L1uGtGenToInputProducer::L1uGtGenToInputProducer(const ParameterSet& iConfig)
  {
    // register what you produce
    produces<BXVector<l1t::EGamma>>();
    produces<BXVector<l1t::Muon>>();
    produces<BXVector<l1t::Tau>>();
    produces<BXVector<l1t::Jet>>();
    produces<BXVector<l1t::EtSum>>();

    // Setup parameters
    bxFirst_ = iConfig.getParameter<int>("bxFirst");
    bxLast_  = iConfig.getParameter<int>("bxLast");

    jetEtThreshold_ = iConfig.getParameter<double>("jetEtThreshold");
    tauEtThreshold_ = iConfig.getParameter<double>("tauEtThreshold");
    egEtThreshold_  = iConfig.getParameter<double>("egEtThreshold");
    muEtThreshold_  = iConfig.getParameter<double>("muEtThreshold");


    genParticlesToken = consumes <reco::GenParticleCollection> (std::string("genParticles"));
    genJetsToken      = consumes <reco::GenJetCollection> (std::string("ak5GenJets"));
    genMetToken       = consumes <reco::GenMETCollection> (std::string("genMetCalo"));


    // set cache id to zero, will be set at first beginRun:
    m_paramsCacheId = 0;
  }


  L1uGtGenToInputProducer::~L1uGtGenToInputProducer()
  {
  }



//
// member functions
//

// ------------ method called to produce the data ------------
void
L1uGtGenToInputProducer::produce(Event& iEvent, const EventSetup& iSetup)
{

  LogDebug("l1t|Global") << "L1uGtGenToInputProducer::produce function called...\n";

  // Setup vectors
  std::vector<l1t::Muon> muonVec;
  std::vector<l1t::EGamma> egammaVec;
  std::vector<l1t::Tau> tauVec;
  std::vector<l1t::Jet> jetVec;
  std::vector<l1t::EtSum> etsumVec;

  // Set the range of BX....TO DO...move to Params or determine from param set.
  int bxFirst = bxFirst_;
  int bxLast  = bxLast_;

  // For now, set bx = 0, update in future
  // int bxEval = 0;

  int maxNumMuCands  = 8;
  int maxNumJetCands = 12;
  int maxNumEGCands  = 12;
  int maxNumTauCands = 8;

  // Default values for EG/Tau. Redefined for individual objects below
  double maxPt_ = 255.;
  int ptSteps_ = 510;
  double minEta_ = -5.;
  double maxEta_ = 5.;
  int etaSteps_ = 230;
  int phiSteps_ = 144;

  //outputs
  std::auto_ptr<l1t::EGammaBxCollection> egammas (new l1t::EGammaBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<l1t::MuonBxCollection> muons (new l1t::MuonBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<l1t::TauBxCollection> taus (new l1t::TauBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<l1t::JetBxCollection> jets (new l1t::JetBxCollection(0, bxFirst, bxLast));
  std::auto_ptr<l1t::EtSumBxCollection> etsums (new l1t::EtSumBxCollection(0, bxFirst, bxLast));

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

      // Only use status 1 particles
      if( status!=1 ) continue;

      int absId = abs(pdgId);

      if( absId==11 && pt>=egEtThreshold_ )       eg_cands_index.push_back(k);
      else if( absId==13 && pt>=muEtThreshold_ )  mu_cands_index.push_back(k);
      else if( absId==15 && pt>=tauEtThreshold_ ) tau_cands_index.push_back(k);
    }
  }
  else {
    LogTrace("l1t|Global") << ">>> GenParticles collection not found!" << std::endl;
  }



  // Muon Collection
  int numMuCands = int( mu_cands_index.size() );
  Int_t idxMu[numMuCands];
  double muPtSorted[numMuCands];
  for( int iMu=0; iMu<numMuCands; iMu++ ) muPtSorted[iMu] = genParticles->at(mu_cands_index[iMu]).pt();

  TMath::Sort(numMuCands,muPtSorted,idxMu);
  for( int iMu=0; iMu<numMuCands; iMu++ ){

    if( iMu>=maxNumMuCands ) continue;

    maxPt_ = 255.;
    ptSteps_ = 510;
    minEta_ = -2.45;
    maxEta_ = 2.45;
    etaSteps_ = 450;
    phiSteps_ = 576;
  
    const reco::Candidate & mcParticle = (*genParticles)[mu_cands_index[idxMu[iMu]]];

    int pt   = convertPtToHW( mcParticle.pt(), maxPt_, ptSteps_ );
    int eta  = convertEtaToHW( mcParticle.eta(), minEta_, maxEta_, etaSteps_, 0x1ff );
    int phi  = convertPhiToHW( mcParticle.phi(), phiSteps_ );
    int qual = 4;
    int iso  = 1;
    int charge = ( mcParticle.charge()>0 ) ? 1 : 0;
    int chargeValid = 1;
    int mip = 1;
    int tag = 1;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

    l1t::Muon mu(*p4, pt, eta, phi, qual, charge, chargeValid, iso, mip, tag);
    //muons->push_back(bxEval, mu);

    muonVec.push_back(mu);
  }


  // Use to sum the energy of the objects in the event for ETT and HTT
  // sum all EG, taus and jets
  double sumEt = 0;

  // EG Collection
  int numEgCands = int( eg_cands_index.size() );
  Int_t idxEg[numEgCands];
  double egPtSorted[numEgCands];
  for( int iEg=0; iEg<numEgCands; iEg++ ) egPtSorted[iEg] = genParticles->at(eg_cands_index[iEg]).pt();

  TMath::Sort(numEgCands,egPtSorted,idxEg);
  for( int iEg=0; iEg<numEgCands; iEg++ ){

    if( iEg>=maxNumEGCands ) continue;

    maxPt_ = 255.;
    ptSteps_ = 510;
    minEta_ = -5.;
    maxEta_ = 5.;
    etaSteps_ = 230;
    phiSteps_ = 144;
  
    const reco::Candidate & mcParticle = (*genParticles)[eg_cands_index[idxEg[iEg]]];

    int pt   = convertPtToHW( mcParticle.pt(), maxPt_, ptSteps_ );
    int eta  = convertEtaToHW( mcParticle.eta(), minEta_, maxEta_, etaSteps_ , 0xff);
    int phi  = convertPhiToHW( mcParticle.phi(), phiSteps_ );
    int qual = 1;
    int iso  = 1;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

    l1t::EGamma eg(*p4, pt, eta, phi, qual, iso);
    //egammas->push_back(bxEval, eg);
    egammaVec.push_back(eg);

    //Keep running sum of total Et
    sumEt += mcParticle.pt();
  }
  


  // Tau Collection
  int numTauCands = int( tau_cands_index.size() );
  Int_t idxTau[numTauCands];
  double tauPtSorted[numTauCands];
  for( int iTau=0; iTau<numTauCands; iTau++ ) tauPtSorted[iTau] = genParticles->at(tau_cands_index[iTau]).pt();

  TMath::Sort(numTauCands,tauPtSorted,idxTau);
  for( int iTau=0; iTau<numTauCands; iTau++ ){

    if( iTau>=maxNumTauCands ) continue;

    maxPt_ = 255.;
    ptSteps_ = 510;
    minEta_ = -5.;
    maxEta_ = 5.;
    etaSteps_ = 230;
    phiSteps_ = 144;
  
    const reco::Candidate & mcParticle = (*genParticles)[tau_cands_index[idxTau[iTau]]];

    int pt   = convertPtToHW( mcParticle.pt(), maxPt_, ptSteps_ );
    int eta  = convertEtaToHW( mcParticle.eta(), minEta_, maxEta_, etaSteps_ , 0xff);
    int phi  = convertPhiToHW( mcParticle.phi(), phiSteps_ );
    int qual = 1;
    int iso  = 1;

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

    l1t::Tau tau(*p4, pt, eta, phi, qual, iso);
    //taus->push_back(bxEval, tau);
    tauVec.push_back(tau);
    
    //Keep running sum of total Et
    sumEt += mcParticle.pt();    
  }


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
      if( nJet>=maxNumJetCands ) continue;
      ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

      int pt  = convertPtToHW( genJet->et(), 1023., 2046 );
      int eta = convertEtaToHW( genJet->eta(), -5., 5., 230, 0xff );
      int phi = convertPhiToHW( genJet->phi(), 144 );

      int qual = 0;

      l1t::Jet jet(*p4, pt, eta, phi, qual);
      //jets->push_back(bxEval, jet);
      jetVec.push_back(jet);

      nJet++;     
    }
  }
  else {
    LogTrace("l1t|Global") << ">>> GenJets collection not found!" << std::endl;
  }


  edm::Handle<reco::GenMETCollection> genMet;
  // Make sure that you can get genMET
  if( iEvent.getByToken(genMetToken, genMet) ){
    int pt  = convertPtToHW( genMet->front().pt(), 2047., 4094 );
    int phi = convertPhiToHW( genMet->front().phi(), 144 );

    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();

    // Missing Et
    l1t::EtSum etmiss(*p4, l1t::EtSum::EtSumType::kMissingEt,pt, 0,phi, 0); 
    //etsums->push_back(bxEval, etmiss);  
    etsumVec.push_back(etmiss);

    // Make Missing Ht slightly smaller and rotated (These are all fake inputs anyway...not supposed to be realistic)
    pt  = convertPtToHW( genMet->front().pt()*0.9, 2047., 4094 );
    phi = convertPhiToHW( genMet->front().phi()+ 3.14/5., 144 );

    l1t::EtSum htmiss(*p4, l1t::EtSum::EtSumType::kMissingHt,pt, 0,phi, 0); 
    //etsums->push_back(bxEval, htmiss);  
    etsumVec.push_back(htmiss);


  }
  else {
    LogTrace("l1t|Global") << ">>> GenMet collection not found!" << std::endl;
  }


// Put the total Et into EtSums  (Make HTT slightly smaller to tell them apart....not supposed to be realistic) 
   int pt  = convertPtToHW( sumEt, 2047., 4094 );
   ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 = new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
   l1t::EtSum etTotal(*p4, l1t::EtSum::EtSumType::kTotalEt,pt, 0, 0, 0); 
   //etsums->push_back(bxEval, etTotal);  
   etsumVec.push_back(etTotal);

   pt  = convertPtToHW( sumEt*0.9, 2047., 4094 );
   l1t::EtSum htTotal(*p4, l1t::EtSum::EtSumType::kTotalHt,pt, 0, 0, 0); 
   //etsums->push_back(bxEval, htTotal);  
   etsumVec.push_back(htTotal);

/*
   counter_++;
   if(counter_==1){
     muonVec_bxm2 = muonVec;
     egammaVec_bxm2 = egammaVec;
     tauVec_bxm2 = tauVec;
     jetVec_bxm2 = jetVec;
     etsumVec_bxm2 = etsumVec;
   }
   else if(counter_==2){
     muonVec_bxm1 = muonVec;
     egammaVec_bxm1 = egammaVec;
     tauVec_bxm1 = tauVec;
     jetVec_bxm1 = jetVec;
     etsumVec_bxm1 = etsumVec;
   }
   else if(counter_==3){
     muonVec_bx0 = muonVec;
     egammaVec_bx0 = egammaVec;
     tauVec_bx0 = tauVec;
     jetVec_bx0 = jetVec;
     etsumVec_bx0 = etsumVec;
   }
   else if( counter_==4 ){
     muonVec_bxp1 = muonVec;
     egammaVec_bxp1 = egammaVec;
     tauVec_bxp1 = tauVec;
     jetVec_bxp1 = jetVec;
     etsumVec_bxp1 = etsumVec;
   }
   else if( counter_==5 ){
*/
 
    // Insert all the bx into the L1 Collections

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
     for( int iMu=0; iMu<int(muonVec.size()); iMu++ ){
       muons->push_back(2, muonVec[iMu]);
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
     for( int iEG=0; iEG<int(egammaVec.size()); iEG++ ){
       egammas->push_back(2, egammaVec[iEG]);
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
     for( int iTau=0; iTau<int(tauVec.size()); iTau++ ){
       taus->push_back(2, tauVec[iTau]);
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
     for( int iJet=0; iJet<int(jetVec.size()); iJet++ ){
       jets->push_back(2, jetVec[iJet]);
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
     for( int iETsum=0; iETsum<int(etsumVec.size()); iETsum++ ){
       etsums->push_back(2, etsumVec[iETsum]);
     }

     // reset counter
//     counter_ = 0;
//   }


  iEvent.put(egammas);
  iEvent.put(muons);
  iEvent.put(taus);
  iEvent.put(jets);
  iEvent.put(etsums);

// Now shift the bx data by one to prepare for next event.
     muonVec_bxm2 = muonVec_bxm1;
     egammaVec_bxm2 = egammaVec_bxm1;
     tauVec_bxm2 = tauVec_bxm1;
     jetVec_bxm2 = jetVec_bxm1;
     etsumVec_bxm2 = etsumVec_bxm1;      

     muonVec_bxm1 = muonVec_bx0;
     egammaVec_bxm1 = egammaVec_bx0;
     tauVec_bxm1 = tauVec_bx0;
     jetVec_bxm1 = jetVec_bx0;
     etsumVec_bxm1 = etsumVec_bx0;      

     muonVec_bx0 = muonVec_bxp1;
     egammaVec_bx0 = egammaVec_bxp1;
     tauVec_bx0 = tauVec_bxp1;
     jetVec_bx0 = jetVec_bxp1;
     etsumVec_bx0 = etsumVec_bxp1;      

     muonVec_bxp1 = muonVec;
     egammaVec_bxp1 = egammaVec;
     tauVec_bxp1 = tauVec;
     jetVec_bxp1 = jetVec;
     etsumVec_bxp1 = etsumVec;   

}

// ------------ method called once each job just before starting event loop ------------
void
L1uGtGenToInputProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop ------------
void
L1uGtGenToInputProducer::endJob() {
}

// ------------ method called when starting to processes a run ------------

void L1uGtGenToInputProducer::beginRun(Run const&iR, EventSetup const&iE){

  LogDebug("l1t|Global") << "L1uGtGenToInputProducer::beginRun function called...\n";

  counter_ = 0;
}

// ------------ method called when ending the processing of a run ------------
void L1uGtGenToInputProducer::endRun(Run const& iR, EventSetup const& iE){

}


// ------------ methods to convert from physical to HW values ------------
int L1uGtGenToInputProducer::convertPhiToHW(double iphi, int steps){

  double phiMax = 2 * M_PI;
  if( iphi < 0 ) iphi += 2*M_PI;
  if( iphi > phiMax) iphi -= phiMax;

  int hwPhi = int( (iphi/phiMax)*steps + 0.00001 );
  return hwPhi;
}

unsigned int L1uGtGenToInputProducer::convertEtaToHW(double ieta, double minEta, double maxEta, int steps, unsigned int bitMask){

  
   double binWidth = (maxEta - minEta)/steps;
     
   //if we are outside the limits (or right on it) ...shift to make sure it is in the range of hwEta
   if(ieta <= minEta) ieta = minEta+binWidth/2.;
   if(ieta >= maxEta) ieta = maxEta-binWidth/2.;
      
   int binNum = (int)(ieta/binWidth);
   if(ieta<0.) binNum--;
      
   unsigned int hwEta = binNum & bitMask; 


  return hwEta;
}

int L1uGtGenToInputProducer::convertPtToHW(double ipt, double maxPt, int steps){

  // if above max Pt, set to largest value
  if(ipt >= maxPt) ipt = maxPt - (maxPt/(2*steps)); 
 
  int hwPt = int( (ipt/maxPt)*steps + 0.00001 );
  
  return hwPt;
}


// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void
L1uGtGenToInputProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

} // namespace

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::L1uGtGenToInputProducer);
