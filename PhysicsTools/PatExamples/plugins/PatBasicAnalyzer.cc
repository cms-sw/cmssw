#include <map>
#include <string>

#include "TH1.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

class PatBasicAnalyzer : public edm::EDAnalyzer {
public:
  /// default constructor
  explicit PatBasicAnalyzer(const edm::ParameterSet&);
  /// default destructor
  ~PatBasicAnalyzer() override;

private:
  /// everything that needs to be done before the event loop
  void beginJob() override;
  /// everything that needs to be done during the event loop
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  /// everything that needs to be done after the event loop
  void endJob() override;

  // simple map to contain all histograms;
  // histograms are booked in the beginJob()
  // method
  std::map<std::string, TH1F*> histContainer_;
  // plot number of towers per jet
  TH1F* jetTowers_;

  // input tokens
  edm::EDGetTokenT<edm::View<pat::Photon>> photonSrcToken_;
  edm::EDGetTokenT<edm::View<pat::Electron>> elecSrcToken_;
  edm::EDGetTokenT<edm::View<pat::Muon>> muonSrcToken_;
  edm::EDGetTokenT<edm::View<pat::Tau>> tauSrcToken_;
  edm::EDGetTokenT<edm::View<pat::Jet>> jetSrcToken_;
  edm::EDGetTokenT<edm::View<pat::MET>> metSrcToken_;
};

PatBasicAnalyzer::PatBasicAnalyzer(const edm::ParameterSet& iConfig)
    : histContainer_(),
      photonSrcToken_(consumes<edm::View<pat::Photon>>(iConfig.getUntrackedParameter<edm::InputTag>("photonSrc"))),
      elecSrcToken_(consumes<edm::View<pat::Electron>>(iConfig.getUntrackedParameter<edm::InputTag>("electronSrc"))),
      muonSrcToken_(consumes<edm::View<pat::Muon>>(iConfig.getUntrackedParameter<edm::InputTag>("muonSrc"))),
      tauSrcToken_(consumes<edm::View<pat::Tau>>(iConfig.getUntrackedParameter<edm::InputTag>("tauSrc"))),
      jetSrcToken_(consumes<edm::View<pat::Jet>>(iConfig.getUntrackedParameter<edm::InputTag>("jetSrc"))),
      metSrcToken_(consumes<edm::View<pat::MET>>(iConfig.getUntrackedParameter<edm::InputTag>("metSrc"))) {}

PatBasicAnalyzer::~PatBasicAnalyzer() {}

void PatBasicAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get electron collection
  edm::Handle<edm::View<pat::Electron>> electrons;
  iEvent.getByToken(elecSrcToken_, electrons);

  // get muon collection
  edm::Handle<edm::View<pat::Muon>> muons;
  iEvent.getByToken(muonSrcToken_, muons);

  // get tau collection
  edm::Handle<edm::View<pat::Tau>> taus;
  iEvent.getByToken(tauSrcToken_, taus);

  // get jet collection
  edm::Handle<edm::View<pat::Jet>> jets;
  iEvent.getByToken(jetSrcToken_, jets);

  // get met collection
  edm::Handle<edm::View<pat::MET>> mets;
  iEvent.getByToken(metSrcToken_, mets);

  // get photon collection
  edm::Handle<edm::View<pat::Photon>> photons;
  iEvent.getByToken(photonSrcToken_, photons);

  // loop over jets
  size_t nJets = 0;
  for (edm::View<pat::Jet>::const_iterator jet = jets->begin(); jet != jets->end(); ++jet) {
    if (jet->pt() > 50) {
      ++nJets;
    }
    // uncomment the following line to fill the
    // jetTowers_ histogram
    // jetTowers_->Fill(jet->getCaloConstituents().size());
  }
  histContainer_["jets"]->Fill(nJets);

  // do something similar for the other candidates
  histContainer_["photons"]->Fill(photons->size());
  histContainer_["elecs"]->Fill(electrons->size());
  histContainer_["muons"]->Fill(muons->size());
  histContainer_["taus"]->Fill(taus->size());
  histContainer_["met"]->Fill(mets->empty() ? 0 : (*mets)[0].et());
}

void PatBasicAnalyzer::beginJob() {
  // register to the TFileService
  edm::Service<TFileService> fs;

  // book histograms:
  // uncomment the following line to book the jetTowers_ histogram
  //jetTowers_= fs->make<TH1F>("jetTowers", "towers per jet",   90, 0,  90);
  histContainer_["photons"] = fs->make<TH1F>("photons", "photon multiplicity", 10, 0, 10);
  histContainer_["elecs"] = fs->make<TH1F>("elecs", "electron multiplicity", 10, 0, 10);
  histContainer_["muons"] = fs->make<TH1F>("muons", "muon multiplicity", 10, 0, 10);
  histContainer_["taus"] = fs->make<TH1F>("taus", "tau multiplicity", 10, 0, 10);
  histContainer_["jets"] = fs->make<TH1F>("jets", "jet multiplicity", 10, 0, 10);
  histContainer_["met"] = fs->make<TH1F>("met", "missing E_{T}", 20, 0, 100);
}

void PatBasicAnalyzer::endJob() {}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PatBasicAnalyzer);
