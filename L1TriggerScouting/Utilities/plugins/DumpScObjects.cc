#include "FWCore/Framework/interface/MakerMacros.h"

#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <cmath>

#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/conversion.h"

using namespace l1ScoutingRun3;

// ----------------------------- CLASS DECLARATION  ----------------------------
class DumpScObjects : public edm::stream::EDAnalyzer<> {

  public:
    // constructor and destructor
    explicit DumpScObjects(const edm::ParameterSet&);
    ~DumpScObjects() override{};

    // method for analyzing the events
    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:

    void printMuon(const ScMuon* muon);
    template <typename T>
    void printCaloObject(const T* obj);
    void printCaloSum(const ScEtSum* sum);

    void printBx(unsigned bx);

    // the tokens to access the data
    edm::EDGetTokenT<ScMuonOrbitCollection>    gmtMuonsToken_;
    edm::EDGetTokenT<ScJetOrbitCollection>     caloJetsToken_;
    edm::EDGetTokenT<ScEGammaOrbitCollection>  caloEGammasToken_;
    edm::EDGetTokenT<ScTauOrbitCollection>     caloTausToken_;
    edm::EDGetTokenT<ScEtSumOrbitCollection>   caloEtSumsToken_;

    edm::Handle<ScMuonOrbitCollection> muonHandle_;
    edm::Handle<ScJetOrbitCollection> jetHandle_;
    edm::Handle<ScEGammaOrbitCollection> eGammaHandle_;
    edm::Handle<ScTauOrbitCollection> tauHandle_;
    edm::Handle<ScEtSumOrbitCollection> etSumHandle_;

    // the min and max BX to be analyzed
    unsigned minBx_;
    unsigned maxBx_;

    // select collection to be printed
    bool checkMuons_;
    bool checkJets_;
    bool checkEGammas_;
    bool checkTaus_;
    bool checkEtSums_;

    // dump a specific (ORBIT, BX RANGE)
    bool searchEvent_;
    unsigned orbitNum_;
    unsigned searchStartBx_;
    unsigned searchStopBx_;

    // utils
    bool skipEmptyBx_;
};
// -----------------------------------------------------------------------------


// -------------------------------- constructor  -------------------------------

DumpScObjects::DumpScObjects(const edm::ParameterSet& iConfig):
  minBx_(iConfig.getUntrackedParameter<unsigned>("minBx", 0)),
  maxBx_(iConfig.getUntrackedParameter<unsigned>("maxBx", 3564)),

  checkMuons_(iConfig.getUntrackedParameter<bool>("checkMuons", true)),
  checkJets_(iConfig.getUntrackedParameter<bool>("checkJets", true)),
  checkEGammas_(iConfig.getUntrackedParameter<bool>("checkEGammas", true)),
  checkTaus_(iConfig.getUntrackedParameter<bool>("checkTaus", true)),
  checkEtSums_(iConfig.getUntrackedParameter<bool>("checkEtSums", true)),

  searchEvent_(iConfig.getUntrackedParameter<bool>("searchEvent", false)),
  orbitNum_(iConfig.getUntrackedParameter<unsigned>("orbitNumber", 0)),
  searchStartBx_(iConfig.getUntrackedParameter<unsigned>("searchStartBx", 0)),
  searchStopBx_(iConfig.getUntrackedParameter<unsigned>("searchStopBx", 0)),

  skipEmptyBx_(iConfig.getUntrackedParameter<bool>("skipEmptyBx", true))
{

  if (checkMuons_) gmtMuonsToken_    = consumes<ScMuonOrbitCollection>(iConfig.getParameter<edm::InputTag>("gmtMuonsTag"));
  if (checkJets_) caloJetsToken_    = consumes<ScJetOrbitCollection>(iConfig.getParameter<edm::InputTag>("caloJetsTag"));
  if (checkEGammas_) caloEGammasToken_ = consumes<ScEGammaOrbitCollection>(iConfig.getParameter<edm::InputTag>("caloEGammasTag"));
  if (checkTaus_) caloTausToken_    = consumes<ScTauOrbitCollection>(iConfig.getParameter<edm::InputTag>("caloTausTag"));
  if (checkEtSums_) caloEtSumsToken_  = consumes<ScEtSumOrbitCollection>(iConfig.getParameter<edm::InputTag>("caloEtSumsTag"));

}
// -----------------------------------------------------------------------------

// ----------------------- method called for each orbit  -----------------------
void DumpScObjects::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {

  if (checkMuons_)   iEvent.getByToken(gmtMuonsToken_, muonHandle_);
  if (checkJets_)    iEvent.getByToken(caloJetsToken_, jetHandle_);
  if (checkEGammas_) iEvent.getByToken(caloEGammasToken_, eGammaHandle_);
  if (checkTaus_)    iEvent.getByToken(caloTausToken_, tauHandle_);
  if (checkEtSums_)  iEvent.getByToken(caloEtSumsToken_, etSumHandle_);

  // get the orbit number
  unsigned currOrbit = iEvent.id().event();

  // if we are looking for a specific orbit
  if (searchEvent_){
    if (currOrbit != orbitNum_) return;
    
    // found the orbit
    for (unsigned bx=searchStartBx_; bx<=searchStopBx_; bx++){
      printBx(bx);
    }
  } else {

    if (skipEmptyBx_){
      
      // create a set of non empty BXs
      std::set<unsigned> uniqueBx;

      if (checkMuons_) {
        for (const unsigned& bx: muonHandle_->getFilledBxs()){
          if ((bx>=minBx_) || (bx<=maxBx_)) uniqueBx.insert(bx);
        }
      }
      if (checkJets_) {
        for (const unsigned& bx: jetHandle_->getFilledBxs()){
          if ((bx>=minBx_) || (bx<=maxBx_)) uniqueBx.insert(bx);
        }
      }
      if (checkEGammas_) {
        for (const unsigned& bx: eGammaHandle_->getFilledBxs()){
          if ((bx>=minBx_) || (bx<=maxBx_)) uniqueBx.insert(bx);
        }
      }
      if (checkTaus_) {
        for (const unsigned& bx: tauHandle_->getFilledBxs()){
          if ((bx>=minBx_) || (bx<=maxBx_)) uniqueBx.insert(bx);
        }
      }
      if (checkEtSums_) {
        for (const unsigned& bx: etSumHandle_->getFilledBxs()){
          if ((bx>=minBx_) || (bx<=maxBx_)) uniqueBx.insert(bx);
        }
      }

      // process bx
      for (const unsigned& bx: uniqueBx){
        printBx(bx);
      }
      
    } 
    else {
      // dump all objects
      for (unsigned bx=minBx_; bx<=maxBx_; bx++){
        printBx(bx);
      }
    }
  }

}
// -----------------------------------------------------------------------------

void DumpScObjects::printMuon(const ScMuon* muon){
  std::cout  <<  "  Pt  [GeV/Hw]: " << ugmt::fPt(muon->hwPt())  << "/" << muon->hwPt() << "\n";
  std::cout  <<  "  Eta [rad/Hw]: " << ugmt::fEta(muon->hwEta()) << "/" << muon->hwEta() << "\n";
  std::cout  <<  "  Phi [rad/Hw]: " << ugmt::fPhi(muon->hwPhi()) << "/" << muon->hwPhi() << "\n";
  std::cout  <<  "  Charge/valid: " << muon->hwCharge() << "/" << muon->hwChargeValid() << "\n";
  std::cout  <<  "  PhiVtx  [rad/Hw]: " << ugmt::fPhiAtVtx(muon->hwPhiAtVtx()) << "/" << muon->hwPhiAtVtx() << "\n";
  std::cout  <<  "  EtaVtx  [rad/Hw]: " << ugmt::fEtaAtVtx(muon->hwEtaAtVtx()) << "/" << muon->hwEtaAtVtx() << "\n";
  std::cout  <<  "  Pt uncon[GeV/Hw]: " << ugmt::fPtUnconstrained(muon->hwPtUnconstrained()) << "/" << muon->hwPtUnconstrained() << "\n";
  std::cout  <<  "  Dxy: "  << muon->hwDXY() << "\n";
  std::cout  <<  "  Qual: " << muon->hwQual() << "\n";
  std::cout  <<  "  TF index: " << muon->tfMuonIndex() << "\n";
}

template <typename T>
void DumpScObjects::printCaloObject(const T* obj){
  std::cout << "  Et  [GeV/Hw]: " << demux::fEt(obj->hwEt())   << "/" << obj->hwEt()  << "\n";
  std::cout << "  Eta [rad/Hw]: " << demux::fEta(obj->hwEta()) << "/" << obj->hwEta() << "\n";
  std::cout << "  Phi [rad/Hw]: " << demux::fPhi(obj->hwPhi()) << "/" << obj->hwPhi() << "\n";
  std::cout << "  Iso [Hw]: " << obj->hwIso() << "\n";
}

void DumpScObjects::printCaloSum(const ScEtSum* sum){

  if (sum->type() == l1t::EtSum::kTotalEt){
    std::cout << "Type: TotalET\n"
              << "  Et [GeV/Hw]: " << demux::fEt(sum->hwEt()) << "/" << sum->hwEt()
              << std::endl;
  }
  if (sum->type() == l1t::EtSum::kTotalHt){
    std::cout << "Type: TotalHT\n"
              << "  Et [GeV/Hw]: " << demux::fEt(sum->hwEt()) << "/" << sum->hwEt()
              << std::endl;
  }

  if (sum->type() == l1t::EtSum::kMissingEt){
    std::cout << "Type: ETMiss\n"
              << "  Et  [GeV/Hw]: " << demux::fEt(sum->hwEt()) << "/" << sum->hwEt() << "\n"
              << "  Phi [Rad/Hw]: " << demux::fPhi(sum->hwPhi()) << "/" << sum->hwPhi()
              << std::endl;
  }

  if (sum->type() == l1t::EtSum::kMissingHt){
    std::cout << "Type: HTMiss\n"
              << "  Et  [GeV/Hw]: " << demux::fEt(sum->hwEt()) << "/" << sum->hwEt() << "\n"
              << "  Phi [Rad/Hw]: " << demux::fPhi(sum->hwPhi()) << "/" << sum->hwPhi()
              << std::endl;
  }
}

void DumpScObjects::printBx(unsigned bx){
  std::cout << "BX = " << bx <<" ****" << std::endl;

  if(checkMuons_ && muonHandle_.isValid()){
    int i=0;
    for (auto muon = muonHandle_->begin(bx); muon!=muonHandle_->end(bx); muon++){
      std::cout  <<  "--- Muon " << i << " ---\n";
      printMuon(&*muon);
      i++;
    }
  }

  if(checkJets_ && jetHandle_.isValid()){
    int i=0;
    for (auto jet = jetHandle_->begin(bx); jet!=jetHandle_->end(bx); jet++){
      std::cout  <<  "--- Jet " << i << " ---\n";
      printCaloObject<ScJet>(&*jet);
      i++;
    }
  }

  if(checkEGammas_ && jetHandle_.isValid()){
    int i=0;
    for (auto egamma = eGammaHandle_->begin(bx); egamma!=eGammaHandle_->end(bx); egamma++){
      std::cout  <<  "--- E/Gamma " << i << " ---\n";
      printCaloObject<ScEGamma>(&*egamma);
      i++;
    }
  }

  if(checkTaus_ && tauHandle_.isValid()){
    int i=0;
    for (auto tau = tauHandle_->begin(bx); tau!=tauHandle_->end(bx); tau++){
      std::cout  <<  "--- Tau " << i << " ---\n";
      printCaloObject<ScTau>(&*tau);
      i++;
    }
  }

  if(checkEtSums_ && etSumHandle_.isValid()){
    int i=0;
    for (auto sum = etSumHandle_->begin(bx); sum!=etSumHandle_->end(bx); sum++){
      std::cout  <<  "--- Sum " << i << " ---\n";
      printCaloSum(&*sum);
      i++;
    }
  }

}

DEFINE_FWK_MODULE(DumpScObjects);