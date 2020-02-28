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
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

class PatTopSelectionAnalyzer : public edm::EDAnalyzer {
public:
  /// default constructor
  explicit PatTopSelectionAnalyzer(const edm::ParameterSet&);
  /// default destructor
  ~PatTopSelectionAnalyzer() override;

private:
  /// everything that needs to be done before the event loop
  void beginJob() override;
  /// everything that needs to be done during the event loop
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  /// everything that needs to be done after the event loop
  void endJob() override;

  /// check if histogram was booked
  bool booked(const std::string histName) const { return hists_.find(histName) != hists_.end(); };
  /// fill histogram if it had been booked before
  void fill(const std::string histName, double value) const {
    if (booked(histName))
      hists_.find(histName)->second->Fill(value);
  };

  // simple map to contain all histograms;
  // histograms are booked in the beginJob()
  // method
  std::map<std::string, TH1F*> hists_;

  // input tags
  edm::EDGetTokenT<edm::View<pat::Electron> > elecsToken_;
  edm::EDGetTokenT<edm::View<pat::Muon> > muonsToken_;
  edm::EDGetTokenT<edm::View<pat::Jet> > jetsToken_;
  edm::EDGetTokenT<edm::View<pat::MET> > metToken_;
};

PatTopSelectionAnalyzer::PatTopSelectionAnalyzer(const edm::ParameterSet& iConfig)
    : hists_(),
      elecsToken_(consumes<edm::View<pat::Electron> >(iConfig.getUntrackedParameter<edm::InputTag>("elecs"))),
      muonsToken_(consumes<edm::View<pat::Muon> >(iConfig.getUntrackedParameter<edm::InputTag>("muons"))),
      jetsToken_(consumes<edm::View<pat::Jet> >(iConfig.getUntrackedParameter<edm::InputTag>("jets"))),
      metToken_(consumes<edm::View<pat::MET> >(iConfig.getUntrackedParameter<edm::InputTag>("met"))) {}

PatTopSelectionAnalyzer::~PatTopSelectionAnalyzer() {}

void PatTopSelectionAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get electron collection
  edm::Handle<edm::View<pat::Electron> > elecs;
  iEvent.getByToken(elecsToken_, elecs);

  // get muon collection
  edm::Handle<edm::View<pat::Muon> > muons;
  iEvent.getByToken(muonsToken_, muons);

  // get jet collection
  edm::Handle<edm::View<pat::Jet> > jets;
  iEvent.getByToken(jetsToken_, jets);

  // get met collection
  edm::Handle<edm::View<pat::MET> > met;
  iEvent.getByToken(metToken_, met);

  // fill yield
  fill("yield", 0.5);

  // fill quantities for leading elec and elec multiplicity
  fill("elecMult", elecs->size());
  if (elecs->begin() != elecs->end()) {
    fill("elecIso", (elecs->begin()->trackIso() + elecs->begin()->caloIso()) / elecs->begin()->pt());
    fill("elecPt", elecs->begin()->pt());
  }

  // fill quantities for leading muon and muon multiplicity
  fill("muonMult", muons->size());
  if (muons->begin() != muons->end()) {
    fill("muonIso", (muons->begin()->trackIso() + muons->begin()->caloIso()) / muons->begin()->pt());
    fill("muonPt", muons->begin()->pt());
  }

  // fill quantities for leading jets and jet multiplicity
  // jet pt is corrected up to L3Absolute
  fill("jetMult", jets->size());
  if (!jets->empty())
    fill("jet0Pt", (*jets)[0].pt());
  if (jets->size() > 1)
    fill("jet1Pt", (*jets)[1].pt());
  if (jets->size() > 2)
    fill("jet2Pt", (*jets)[2].pt());
  if (jets->size() > 3)
    fill("jet3Pt", (*jets)[3].pt());

  // fill MET
  fill("met", met->empty() ? 0 : (*met)[0].et());
}

void PatTopSelectionAnalyzer::beginJob() {
  // register to the TFileService
  edm::Service<TFileService> fs;

  // book histograms:
  hists_["yield"] = fs->make<TH1F>("yield", "electron multiplicity", 1, 0., 1.);
  hists_["elecMult"] = fs->make<TH1F>("elecMult", "electron multiplicity", 10, 0., 10.);
  hists_["elecIso"] = fs->make<TH1F>("elecIso", "electron isolation", 20, 0., 1.);
  hists_["elecPt"] = fs->make<TH1F>("elecPt", "electron pt", 30, 0., 150.);
  hists_["muonMult"] = fs->make<TH1F>("muonMult", "muon multiplicity", 10, 0., 10.);
  hists_["muonIso"] = fs->make<TH1F>("muonIso", "muon isolation", 20, 0., 1.);
  hists_["muonPt"] = fs->make<TH1F>("muonPt", "muon pt", 30, 0., 150.);
  hists_["jetMult"] = fs->make<TH1F>("jetMult", "jet multiplicity", 15, 0., 15.);
  hists_["jet0Pt"] = fs->make<TH1F>("jet0Pt", "1. leading jet pt", 50, 0., 250.);
  hists_["jet1Pt"] = fs->make<TH1F>("jet1Pt", "1. leading jet pt", 50, 0., 250.);
  hists_["jet2Pt"] = fs->make<TH1F>("jet2Pt", "1. leading jet pt", 50, 0., 200.);
  hists_["jet3Pt"] = fs->make<TH1F>("jet3Pt", "1. leading jet pt", 50, 0., 200.);
  hists_["met"] = fs->make<TH1F>("met", "missing E_{T}", 25, 0., 200.);
}

void PatTopSelectionAnalyzer::endJob() {}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PatTopSelectionAnalyzer);
