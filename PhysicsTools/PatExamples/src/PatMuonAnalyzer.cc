#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "PhysicsTools/PatExamples/interface/PatMuonAnalyzer.h"


/// default constructor
PatMuonAnalyzer::PatMuonAnalyzer(const edm::ParameterSet& cfg, TFileDirectory& fs):
  edm::BasicAnalyzer::BasicAnalyzer(cfg, fs),
  muons_(cfg.getParameter<edm::InputTag>("muons"))
{
  hists_["muonPt"  ] = fs.make<TH1F>("muonPt"  , "pt"  ,  100,  0., 300.);
  hists_["muonEta" ] = fs.make<TH1F>("muonEta" , "eta" ,  100, -3.,   3.);
  hists_["muonPhi" ] = fs.make<TH1F>("muonPhi" , "phi" ,  100, -5.,   5.);
}
PatMuonAnalyzer::PatMuonAnalyzer(const edm::ParameterSet& cfg, TFileDirectory& fs, edm::ConsumesCollector&& iC):
  edm::BasicAnalyzer::BasicAnalyzer(cfg, fs),
  muons_(cfg.getParameter<edm::InputTag>("muons"))  ,
  muonsToken_(iC.consumes<std::vector<pat::Muon> >(muons_))
{
  hists_["muonPt"  ] = fs.make<TH1F>("muonPt"  , "pt"  ,  100,  0., 300.);
  hists_["muonEta" ] = fs.make<TH1F>("muonEta" , "eta" ,  100, -3.,   3.);
  hists_["muonPhi" ] = fs.make<TH1F>("muonPhi" , "phi" ,  100, -5.,   5.);
}

/// everything that needs to be done during the event loop
void
PatMuonAnalyzer::analyze(const edm::EventBase& event)
{
  // define what muon you are using; this is necessary as FWLite is not
  // capable of reading edm::Views
  using pat::Muon;

  // Handle to the muon collection
  edm::Handle<std::vector<Muon> > muons;
  event.getByLabel(muons_, muons);

  // loop muon collection and fill histograms
  for(std::vector<Muon>::const_iterator mu1=muons->begin(); mu1!=muons->end(); ++mu1){
    hists_["muonPt" ]->Fill( mu1->pt () );
    hists_["muonEta"]->Fill( mu1->eta() );
    hists_["muonPhi"]->Fill( mu1->phi() );
  }
}
