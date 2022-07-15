///
/// \class l1t::BXVectorInputProducer
///
/// Description: Create Proper BX Vector Structure for full GT Test Vector Generation including out of time BX.
///
///              The producer takes the CAL collections with BX=0 and shifts them, inserting them at BX = -2
///              and ratcheting older BX information through BX = -1, 0, 1, 2.
///
///
/// \author: B Winer OSU
///
///  Modeled after GenToInputProducer.cc

// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
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
#include "DataFormats/L1Trigger/interface/MuonShower.h"
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

  class BXVectorInputProducer : public one::EDProducer<> {
  public:
    explicit BXVectorInputProducer(const ParameterSet&);
    ~BXVectorInputProducer() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void produce(Event&, EventSetup const&) override;

    int convertPhiToHW(double iphi, int steps) const;
    int convertEtaToHW(double ieta, double minEta, double maxEta, int steps) const;
    int convertPtToHW(double ipt, int maxPt, double step) const;

    // ----------member data ---------------------------
    //std::shared_ptr<const CaloParams> m_dbpars; // Database parameters for the trigger, to be updated as needed.
    //std::shared_ptr<const FirmwareVersion> m_fwv;
    //std::shared_ptr<FirmwareVersion> m_fwv; //not const during testing.

    // BX parameters
    int bxFirst_;
    int bxLast_;

    unsigned int maxNumMuCands_;
    unsigned int maxNumMuShowerCands_;
    unsigned int maxNumJetCands_;
    unsigned int maxNumEGCands_;
    unsigned int maxNumTauCands_;

    double jetEtThreshold_;
    double tauEtThreshold_;
    double egEtThreshold_;
    double muEtThreshold_;

    // Control how to end the job
    int emptyBxTrailer_;
    int emptyBxEvt_;
    int eventCnt_;

    // Tokens for inputs from other parts of the L1 system
    edm::EDGetToken egToken;
    edm::EDGetToken muToken;
    edm::EDGetToken muShowerToken;
    edm::EDGetToken tauToken;
    edm::EDGetToken jetToken;
    edm::EDGetToken etsumToken;

    std::vector<l1t::Muon> muonVec_bxm2;
    std::vector<l1t::Muon> muonVec_bxm1;
    std::vector<l1t::Muon> muonVec_bx0;
    std::vector<l1t::Muon> muonVec_bxp1;

    std::vector<l1t::MuonShower> muonShowerVec_bxm2;
    std::vector<l1t::MuonShower> muonShowerVec_bxm1;
    std::vector<l1t::MuonShower> muonShowerVec_bx0;
    std::vector<l1t::MuonShower> muonShowerVec_bxp1;

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
  BXVectorInputProducer::BXVectorInputProducer(const ParameterSet& iConfig) {
    egToken = consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<InputTag>("egInputTag"));
    muToken = consumes<BXVector<l1t::Muon>>(iConfig.getParameter<InputTag>("muInputTag"));
    muShowerToken = consumes<BXVector<l1t::MuonShower>>(iConfig.getParameter<InputTag>("muShowerInputTag"));
    tauToken = consumes<BXVector<l1t::Tau>>(iConfig.getParameter<InputTag>("tauInputTag"));
    jetToken = consumes<BXVector<l1t::Jet>>(iConfig.getParameter<InputTag>("jetInputTag"));
    etsumToken = consumes<BXVector<l1t::EtSum>>(iConfig.getParameter<InputTag>("etsumInputTag"));

    // register what you produce
    produces<BXVector<l1t::EGamma>>();
    produces<BXVector<l1t::Muon>>();
    produces<BXVector<l1t::MuonShower>>();
    produces<BXVector<l1t::Tau>>();
    produces<BXVector<l1t::Jet>>();
    produces<BXVector<l1t::EtSum>>();

    // Setup parameters
    bxFirst_ = iConfig.getParameter<int>("bxFirst");
    bxLast_ = iConfig.getParameter<int>("bxLast");

    maxNumMuCands_ = iConfig.getParameter<unsigned int>("maxMuCand");
    maxNumMuShowerCands_ = iConfig.getParameter<unsigned int>("maxMuShowerCand");
    maxNumJetCands_ = iConfig.getParameter<unsigned int>("maxJetCand");
    maxNumEGCands_ = iConfig.getParameter<unsigned int>("maxEGCand");
    maxNumTauCands_ = iConfig.getParameter<unsigned int>("maxTauCand");

    jetEtThreshold_ = iConfig.getParameter<double>("jetEtThreshold");
    tauEtThreshold_ = iConfig.getParameter<double>("tauEtThreshold");
    egEtThreshold_ = iConfig.getParameter<double>("egEtThreshold");
    muEtThreshold_ = iConfig.getParameter<double>("muEtThreshold");

    emptyBxTrailer_ = iConfig.getParameter<int>("emptyBxTrailer");
    emptyBxEvt_ = iConfig.getParameter<int>("emptyBxEvt");

    // set cache id to zero, will be set at first beginRun:
    eventCnt_ = 0;
  }

  BXVectorInputProducer::~BXVectorInputProducer() {}

  //
  // member functions
  //

  // ------------ method called to produce the data ------------
  void BXVectorInputProducer::produce(Event& iEvent, const EventSetup& iSetup) {
    eventCnt_++;

    LogDebug("l1t|Global") << "BXVectorInputProducer::produce function called...\n";

    // Setup vectors
    std::vector<l1t::Muon> muonVec;
    std::vector<l1t::MuonShower> muonShowerVec;
    std::vector<l1t::EGamma> egammaVec;
    std::vector<l1t::Tau> tauVec;
    std::vector<l1t::Jet> jetVec;
    std::vector<l1t::EtSum> etsumVec;

    // Set the range of BX....TO DO...move to Params or determine from param set.
    int bxFirst = bxFirst_;
    int bxLast = bxLast_;

    //outputs
    std::unique_ptr<l1t::EGammaBxCollection> egammas(new l1t::EGammaBxCollection(0, bxFirst, bxLast));
    std::unique_ptr<l1t::MuonBxCollection> muons(new l1t::MuonBxCollection(0, bxFirst, bxLast));
    std::unique_ptr<l1t::MuonShowerBxCollection> muonShowers(new l1t::MuonShowerBxCollection(0, bxFirst, bxLast));
    std::unique_ptr<l1t::TauBxCollection> taus(new l1t::TauBxCollection(0, bxFirst, bxLast));
    std::unique_ptr<l1t::JetBxCollection> jets(new l1t::JetBxCollection(0, bxFirst, bxLast));
    std::unique_ptr<l1t::EtSumBxCollection> etsums(new l1t::EtSumBxCollection(0, bxFirst, bxLast));

    std::vector<int> mu_cands_index;
    std::vector<int> eg_cands_index;
    std::vector<int> tau_cands_index;

    // Bx to use...grab only bx=0 for now
    int bx = 0;

    // Make sure that you can get input EG
    Handle<BXVector<l1t::EGamma>> inputEgammas;
    if (iEvent.getByToken(egToken, inputEgammas)) {
      for (std::vector<l1t::EGamma>::const_iterator eg = inputEgammas->begin(bx); eg != inputEgammas->end(bx); ++eg) {
        if (eg->hwPt() > egEtThreshold_ && egammaVec.size() < maxNumEGCands_) {
          egammaVec.push_back((*eg));
        }
      }
    } else {
      LogTrace("l1t|Global") << ">>> input EG collection not found!" << std::endl;
    }

    // Make sure that you can get input Muons
    Handle<BXVector<l1t::Muon>> inputMuons;
    if (iEvent.getByToken(muToken, inputMuons)) {
      for (std::vector<l1t::Muon>::const_iterator mu = inputMuons->begin(bx); mu != inputMuons->end(bx); ++mu) {
        if (mu->hwPt() > muEtThreshold_ && muonVec.size() < maxNumMuCands_) {
          muonVec.push_back((*mu));
        }
      }
    } else {
      LogTrace("l1t|Global") << ">>> input Mu collection not found!" << std::endl;
    }

    // Make sure that you can get input Muon Showers
    Handle<BXVector<l1t::MuonShower>> inputMuonShowers;
    if (iEvent.getByToken(muToken, inputMuonShowers)) {
      for (std::vector<l1t::MuonShower>::const_iterator mu = inputMuonShowers->begin(bx);
           mu != inputMuonShowers->end(bx);
           ++mu) {
        if (mu->isValid() && muonShowerVec.size() < maxNumMuCands_) {
          muonShowerVec.push_back((*mu));
        }
      }
    } else {
      LogTrace("l1t|Global") << ">>> input Mu collection not found!" << std::endl;
    }

    // Make sure that you can get input Tau
    Handle<BXVector<l1t::Tau>> inputTaus;
    if (iEvent.getByToken(tauToken, inputTaus)) {
      for (std::vector<l1t::Tau>::const_iterator tau = inputTaus->begin(bx); tau != inputTaus->end(bx); ++tau) {
        if (tau->hwPt() > tauEtThreshold_ && tauVec.size() < maxNumTauCands_) {
          tauVec.push_back((*tau));
        }
      }
    } else {
      LogTrace("l1t|Global") << ">>> input tau collection not found!" << std::endl;
    }

    // Make sure that you can get input jet
    Handle<BXVector<l1t::Jet>> inputJets;
    if (iEvent.getByToken(jetToken, inputJets)) {
      for (std::vector<l1t::Jet>::const_iterator jet = inputJets->begin(bx); jet != inputJets->end(bx); ++jet) {
        if (jet->hwPt() > jetEtThreshold_ && jetVec.size() < maxNumJetCands_) {
          jetVec.push_back((*jet));
        }
      }
    } else {
      LogTrace("l1t|Global") << ">>> input jet collection not found!" << std::endl;
    }

    // Make sure that you can get input etsum
    Handle<BXVector<l1t::EtSum>> inputEtsums;
    if (iEvent.getByToken(etsumToken, inputEtsums)) {
      for (std::vector<l1t::EtSum>::const_iterator etsum = inputEtsums->begin(bx); etsum != inputEtsums->end(bx);
           ++etsum) {
        etsumVec.push_back((*etsum));
      }
    } else {
      LogTrace("l1t|Global") << ">>> input etsum collection not found!" << std::endl;
    }

    // Insert all the bx into the L1 Collections
    LogTrace("l1t|Global") << "Event " << eventCnt_ << " EmptyBxEvt " << emptyBxEvt_ << " emptyBxTrailer "
                           << emptyBxTrailer_ << " diff " << (emptyBxEvt_ - eventCnt_) << std::endl;

    // Fill Muons
    for (int iMu = 0; iMu < int(muonVec_bxm2.size()); iMu++) {
      muons->push_back(-2, muonVec_bxm2[iMu]);
    }
    for (int iMu = 0; iMu < int(muonVec_bxm1.size()); iMu++) {
      muons->push_back(-1, muonVec_bxm1[iMu]);
    }
    for (int iMu = 0; iMu < int(muonVec_bx0.size()); iMu++) {
      muons->push_back(0, muonVec_bx0[iMu]);
    }
    for (int iMu = 0; iMu < int(muonVec_bxp1.size()); iMu++) {
      muons->push_back(1, muonVec_bxp1[iMu]);
    }
    if (emptyBxTrailer_ <= (emptyBxEvt_ - eventCnt_)) {
      for (int iMu = 0; iMu < int(muonVec.size()); iMu++) {
        muons->push_back(2, muonVec[iMu]);
      }
    } else {
      // this event is part of empty trailer...clear out data
      muonVec.clear();
    }

    // Fill MuonShowers
    for (int iMuShower = 0; iMuShower < int(muonShowerVec_bxm2.size()); iMuShower++) {
      muonShowers->push_back(-2, muonShowerVec_bxm2[iMuShower]);
    }
    for (int iMuShower = 0; iMuShower < int(muonShowerVec_bxm1.size()); iMuShower++) {
      muonShowers->push_back(-1, muonShowerVec_bxm1[iMuShower]);
    }
    for (int iMuShower = 0; iMuShower < int(muonShowerVec_bx0.size()); iMuShower++) {
      muonShowers->push_back(0, muonShowerVec_bx0[iMuShower]);
    }
    for (int iMuShower = 0; iMuShower < int(muonShowerVec_bxp1.size()); iMuShower++) {
      muonShowers->push_back(1, muonShowerVec_bxp1[iMuShower]);
    }
    if (emptyBxTrailer_ <= (emptyBxEvt_ - eventCnt_)) {
      for (int iMuShower = 0; iMuShower < int(muonShowerVec.size()); iMuShower++) {
        muonShowers->push_back(2, muonShowerVec[iMuShower]);
      }
    } else {
      // this event is part of empty trailer...clear out data
      muonShowerVec.clear();
    }

    // Fill Egammas
    for (int iEG = 0; iEG < int(egammaVec_bxm2.size()); iEG++) {
      egammas->push_back(-2, egammaVec_bxm2[iEG]);
    }
    for (int iEG = 0; iEG < int(egammaVec_bxm1.size()); iEG++) {
      egammas->push_back(-1, egammaVec_bxm1[iEG]);
    }
    for (int iEG = 0; iEG < int(egammaVec_bx0.size()); iEG++) {
      egammas->push_back(0, egammaVec_bx0[iEG]);
    }
    for (int iEG = 0; iEG < int(egammaVec_bxp1.size()); iEG++) {
      egammas->push_back(1, egammaVec_bxp1[iEG]);
    }
    if (emptyBxTrailer_ <= (emptyBxEvt_ - eventCnt_)) {
      for (int iEG = 0; iEG < int(egammaVec.size()); iEG++) {
        egammas->push_back(2, egammaVec[iEG]);
      }
    } else {
      // this event is part of empty trailer...clear out data
      egammaVec.clear();
    }

    // Fill Taus
    for (int iTau = 0; iTau < int(tauVec_bxm2.size()); iTau++) {
      taus->push_back(-2, tauVec_bxm2[iTau]);
    }
    for (int iTau = 0; iTau < int(tauVec_bxm1.size()); iTau++) {
      taus->push_back(-1, tauVec_bxm1[iTau]);
    }
    for (int iTau = 0; iTau < int(tauVec_bx0.size()); iTau++) {
      taus->push_back(0, tauVec_bx0[iTau]);
    }
    for (int iTau = 0; iTau < int(tauVec_bxp1.size()); iTau++) {
      taus->push_back(1, tauVec_bxp1[iTau]);
    }
    if (emptyBxTrailer_ <= (emptyBxEvt_ - eventCnt_)) {
      for (int iTau = 0; iTau < int(tauVec.size()); iTau++) {
        taus->push_back(2, tauVec[iTau]);
      }
    } else {
      // this event is part of empty trailer...clear out data
      tauVec.clear();
    }

    // Fill Jets
    for (int iJet = 0; iJet < int(jetVec_bxm2.size()); iJet++) {
      jets->push_back(-2, jetVec_bxm2[iJet]);
    }
    for (int iJet = 0; iJet < int(jetVec_bxm1.size()); iJet++) {
      jets->push_back(-1, jetVec_bxm1[iJet]);
    }
    for (int iJet = 0; iJet < int(jetVec_bx0.size()); iJet++) {
      jets->push_back(0, jetVec_bx0[iJet]);
    }
    for (int iJet = 0; iJet < int(jetVec_bxp1.size()); iJet++) {
      jets->push_back(1, jetVec_bxp1[iJet]);
    }
    if (emptyBxTrailer_ <= (emptyBxEvt_ - eventCnt_)) {
      for (int iJet = 0; iJet < int(jetVec.size()); iJet++) {
        jets->push_back(2, jetVec[iJet]);
      }
    } else {
      // this event is part of empty trailer...clear out data
      jetVec.clear();
    }

    // Fill Etsums
    for (int iETsum = 0; iETsum < int(etsumVec_bxm2.size()); iETsum++) {
      etsums->push_back(-2, etsumVec_bxm2[iETsum]);
    }
    for (int iETsum = 0; iETsum < int(etsumVec_bxm1.size()); iETsum++) {
      etsums->push_back(-1, etsumVec_bxm1[iETsum]);
    }
    for (int iETsum = 0; iETsum < int(etsumVec_bx0.size()); iETsum++) {
      etsums->push_back(0, etsumVec_bx0[iETsum]);
    }
    for (int iETsum = 0; iETsum < int(etsumVec_bxp1.size()); iETsum++) {
      etsums->push_back(1, etsumVec_bxp1[iETsum]);
    }
    if (emptyBxTrailer_ <= (emptyBxEvt_ - eventCnt_)) {
      for (int iETsum = 0; iETsum < int(etsumVec.size()); iETsum++) {
        etsums->push_back(2, etsumVec[iETsum]);
      }
    } else {
      // this event is part of empty trailer...clear out data
      etsumVec.clear();
    }

    iEvent.put(std::move(egammas));
    iEvent.put(std::move(muons));
    iEvent.put(std::move(muonShowers));
    iEvent.put(std::move(taus));
    iEvent.put(std::move(jets));
    iEvent.put(std::move(etsums));

    // Now shift the bx data by one to prepare for next event.
    muonVec_bxm2 = muonVec_bxm1;
    muonShowerVec_bxm2 = muonShowerVec_bxm1;
    egammaVec_bxm2 = egammaVec_bxm1;
    tauVec_bxm2 = tauVec_bxm1;
    jetVec_bxm2 = jetVec_bxm1;
    etsumVec_bxm2 = etsumVec_bxm1;

    muonVec_bxm1 = muonVec_bx0;
    muonShowerVec_bxm1 = muonShowerVec_bx0;
    egammaVec_bxm1 = egammaVec_bx0;
    tauVec_bxm1 = tauVec_bx0;
    jetVec_bxm1 = jetVec_bx0;
    etsumVec_bxm1 = etsumVec_bx0;

    muonVec_bx0 = muonVec_bxp1;
    muonShowerVec_bx0 = muonShowerVec_bxp1;
    egammaVec_bx0 = egammaVec_bxp1;
    tauVec_bx0 = tauVec_bxp1;
    jetVec_bx0 = jetVec_bxp1;
    etsumVec_bx0 = etsumVec_bxp1;

    muonVec_bxp1 = muonVec;
    muonShowerVec_bxp1 = muonShowerVec;
    egammaVec_bxp1 = egammaVec;
    tauVec_bxp1 = tauVec;
    jetVec_bxp1 = jetVec;
    etsumVec_bxp1 = etsumVec;
  }

  // ------------ methods to convert from physical to HW values ------------
  int BXVectorInputProducer::convertPhiToHW(double iphi, int steps) const {
    double phiMax = 2 * M_PI;
    if (iphi < 0)
      iphi += 2 * M_PI;
    if (iphi > phiMax)
      iphi -= phiMax;

    int hwPhi = int((iphi / phiMax) * steps + 0.00001);
    return hwPhi;
  }

  int BXVectorInputProducer::convertEtaToHW(double ieta, double minEta, double maxEta, int steps) const {
    double binWidth = (maxEta - minEta) / steps;

    //if we are outside the limits, set error
    if (ieta < minEta)
      return 99999;  //ieta = minEta+binWidth/2.;
    if (ieta > maxEta)
      return 99999;  //ieta = maxEta-binWidth/2.;

    int binNum = (int)(ieta / binWidth);
    if (ieta < 0.)
      binNum--;

    //   unsigned int hwEta = binNum & bitMask;
    //   Remove masking for BXVectors...only assume in raw data

    return binNum;
  }

  int BXVectorInputProducer::convertPtToHW(double ipt, int maxPt, double step) const {
    int hwPt = int(ipt / step + 0.0001);
    // if above max Pt, set to largest value
    if (hwPt > maxPt)
      hwPt = maxPt;

    return hwPt;
  }

  // ------------ method fills 'descriptions' with the allowed parameters for the module ------------
  void BXVectorInputProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
    //The following says we do not know what parameters are allowed so do no validation
    // Please change this to state exactly what you do use, even if it is no parameters
    ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
  }

}  // namespace l1t

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::BXVectorInputProducer);
