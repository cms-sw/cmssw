#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "PhysicsTools/PatExamples/interface/BasicMuonAnalyzer.h"


/// default constructor
BasicMuonAnalyzer::BasicMuonAnalyzer(const edm::ParameterSet& cfg, TFileDirectory& fs): 
  edm::BasicAnalyzer::BasicAnalyzer(cfg, fs),
  muons_(cfg.getParameter<edm::InputTag>("muons"))
{
  hists_["muonPt" ] = fs.make<TH1F>("muonPt", "pt",    100,  0.,300.);
  hists_["muonEta"] = fs.make<TH1F>("muonEta","eta",   100, -3.,  3.);
  hists_["muonPhi"] = fs.make<TH1F>("muonPhi","phi",   100, -5.,  5.); 
}

/// everything that needs to be done during the event loop
void 
BasicMuonAnalyzer::analyze(const edm::EventBase& event)
{
  // Handle to the muon collection
  edm::Handle<std::vector<pat::Muon> > muons;
  event.getByLabel(muons_, muons);

  // loop muon collection and fill histograms
  for(unsigned i=0; i<muons->size(); ++i){
    hists_["muonPt" ]->Fill( (*muons)[i].pt()  );
    hists_["muonEta"]->Fill( (*muons)[i].eta() );
    hists_["muonPhi"]->Fill( (*muons)[i].phi() );
  }
}
