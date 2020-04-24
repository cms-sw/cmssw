#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "PhysicsTools/UtilAlgos/interface/BasicMuonAnalyzer.h"


/// default constructor
BasicMuonAnalyzer::BasicMuonAnalyzer(const edm::ParameterSet& cfg, TFileDirectory& fs):
  edm::BasicAnalyzer::BasicAnalyzer(cfg, fs),
  muons_(cfg.getParameter<edm::InputTag>("muons"))
{
  hists_["muonPt"  ] = fs.make<TH1F>("muonPt"  , "pt"  ,  100,  0., 300.);
  hists_["muonEta" ] = fs.make<TH1F>("muonEta" , "eta" ,  100, -3.,   3.);
  hists_["muonPhi" ] = fs.make<TH1F>("muonPhi" , "phi" ,  100, -5.,   5.);
  hists_["mumuMass"] = fs.make<TH1F>("mumuMass", "mass",   90, 30., 120.);
}
BasicMuonAnalyzer::BasicMuonAnalyzer(const edm::ParameterSet& cfg, TFileDirectory& fs, edm::ConsumesCollector&& iC):
  edm::BasicAnalyzer::BasicAnalyzer(cfg, fs),
  muons_(cfg.getParameter<edm::InputTag>("muons"))  ,
  muonsToken_(iC.consumes<std::vector<reco::Muon> >(muons_))
{
  hists_["muonPt"  ] = fs.make<TH1F>("muonPt"  , "pt"  ,  100,  0., 300.);
  hists_["muonEta" ] = fs.make<TH1F>("muonEta" , "eta" ,  100, -3.,   3.);
  hists_["muonPhi" ] = fs.make<TH1F>("muonPhi" , "phi" ,  100, -5.,   5.);
  hists_["mumuMass"] = fs.make<TH1F>("mumuMass", "mass",   90, 30., 120.);
}

/// everything that needs to be done during the event loop
void
BasicMuonAnalyzer::analyze(const edm::EventBase& event)
{
  // define what muon you are using; this is necessary as FWLite is not
  // capable of reading edm::Views
  using reco::Muon;

  // Handle to the muon collection
  edm::Handle<std::vector<Muon> > muons;
  event.getByLabel(muons_, muons);

  // loop muon collection and fill histograms
  for(std::vector<Muon>::const_iterator mu1=muons->begin(); mu1!=muons->end(); ++mu1){
    hists_["muonPt" ]->Fill( mu1->pt () );
    hists_["muonEta"]->Fill( mu1->eta() );
    hists_["muonPhi"]->Fill( mu1->phi() );
    if( mu1->pt()>20 && fabs(mu1->eta())<2.1 ){
      for(std::vector<Muon>::const_iterator mu2=muons->begin(); mu2!=muons->end(); ++mu2){
	if(mu2>mu1){ // prevent double conting
	  if( mu1->charge()*mu2->charge()<0 ){ // check only muon pairs of unequal charge
	    if( mu2->pt()>20 && fabs(mu2->eta())<2.1 ){
	      hists_["mumuMass"]->Fill( (mu1->p4()+mu2->p4()).mass() );
	    }
	  }
	}
      }
    }
  }
}
