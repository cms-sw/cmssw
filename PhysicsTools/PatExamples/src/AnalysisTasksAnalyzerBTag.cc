#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "PhysicsTools/PatExamples/interface/AnalysisTasksAnalyzerBTag.h"


/// default constructor
AnalysisTasksAnalyzerBTag::AnalysisTasksAnalyzerBTag(const edm::ParameterSet& cfg, TFileDirectory& fs): 
  edm::BasicAnalyzer::BasicAnalyzer(cfg, fs),
  Jets_(cfg.getParameter<edm::InputTag>("Jets")),
  bTagAlgo_(cfg.getParameter<std::string>("bTagAlgo")),
  bins_(cfg.getParameter<unsigned int>("bins")),
  lowerbin_(cfg.getParameter<double>("lowerbin")),
  upperbin_(cfg.getParameter<double>("upperbin"))
{
  hists_["BTag_b"] = fs.make<TH1F>("BTag_b"  , "BTag_b"  ,  bins_,  lowerbin_, upperbin_);
  hists_["BTag_g"] = fs.make<TH1F>("BTag_g" , "BTag_g" ,  bins_, lowerbin_,   upperbin_);
  hists_["BTag_c"] = fs.make<TH1F>("BTag_c" , "BTag_c" ,  bins_, lowerbin_,   upperbin_); 
  hists_["BTag_uds"] = fs.make<TH1F>("BTag_uds", "BTag_uds",   bins_, lowerbin_, upperbin_);
  hists_["BTag_other"] = fs.make<TH1F>("BTag_other", "BTag_other",   bins_, lowerbin_, upperbin_);
  hists_["effBTag_b"] = fs.make<TH1F>("effBTag_b"  , "effBTag_b"  ,  bins_,  lowerbin_, upperbin_);
  hists_["effBTag_g"] = fs.make<TH1F>("effBTag_g" , "effBTag_g" ,  bins_, lowerbin_,  upperbin_);
  hists_["effBTag_c"] = fs.make<TH1F>("effBTag_c" , "effBTag_c" ,  bins_, lowerbin_, upperbin_); 
  hists_["effBTag_uds"] = fs.make<TH1F>("effBTag_uds", "effBTag_uds",   bins_, lowerbin_, upperbin_);
  hists_["effBTag_other"] = fs.make<TH1F>("effBTag_other", "effBTag_other",  bins_, lowerbin_, upperbin_);
}
AnalysisTasksAnalyzerBTag::~AnalysisTasksAnalyzerBTag()
{
  for(unsigned int i=0; i< bins_; ++i){
   hists_["effBTag_b"]->SetBinContent(i,hists_["BTag_b"]->Integral(i,hists_["BTag_b"]->GetNbinsX()+1)/hists_["BTag_b"]->Integral(0,hists_["BTag_b"]->GetNbinsX()+1) );
   hists_["effBTag_g"]->SetBinContent(i,hists_["BTag_g"]->Integral(i,hists_["BTag_g"]->GetNbinsX()+1)/hists_["BTag_g"]->Integral(0,hists_["BTag_g"]->GetNbinsX()+1) );
  hists_["effBTag_c"]->SetBinContent(i,hists_["BTag_c"]->Integral(i,hists_["BTag_c"]->GetNbinsX()+1)/hists_["BTag_c"]->Integral(0,hists_["BTag_c"]->GetNbinsX()+1) );
  hists_["effBTag_uds"]->SetBinContent(i,hists_["BTag_uds"]->Integral(i,hists_["BTag_uds"]->GetNbinsX()+1)/hists_["BTag_uds"]->Integral(0,hists_["BTag_uds"]->GetNbinsX()+1) );
  hists_["effBTag_other"]->SetBinContent(i,hists_["BTag_other"]->Integral(i,hists_["BTag_other"]->GetNbinsX()+1)/hists_["BTag_other"]->Integral(0,hists_["BTag_other"]->GetNbinsX()+1) );
  } 
}
/// everything that needs to be done during the event loop
void 
AnalysisTasksAnalyzerBTag::analyze(const edm::EventBase& event)
{
  // define what Jet you are using; this is necessary as FWLite is not 
  // capable of reading edm::Views
  using pat::Jet;

  // Handle to the Jet collection
  edm::Handle<std::vector<Jet> > Jets;
  event.getByLabel(Jets_, Jets);

  // loop Jet collection and fill histograms
  for(std::vector<Jet>::const_iterator Jet_it=Jets->begin(); Jet_it!=Jets->end(); ++Jet_it){
  
    pat::Jet Jet(*Jet_it);

   //Categorize the Jets
    if( abs(Jet.partonFlavour())==5){
      hists_["BTag_b"]->Fill(Jet.bDiscriminator(bTagAlgo_));
    }
    else{ 
      if( abs(Jet.partonFlavour())==21 || abs(Jet.partonFlavour())==9 ){
	hists_["BTag_g"]->Fill(Jet.bDiscriminator(bTagAlgo_));
      }
      else{
	if( abs(Jet.partonFlavour())==4){
	  hists_["BTag_c"]->Fill(Jet.bDiscriminator(bTagAlgo_));
	}
	else{
	  if( abs(Jet.partonFlavour())==1 || abs(Jet.partonFlavour())==2 || abs(Jet.partonFlavour())==3){
	    hists_["BTag_uds"]->Fill(Jet.bDiscriminator(bTagAlgo_));
	  }
	  else{
	    hists_["BTag_other"]->Fill(Jet.bDiscriminator(bTagAlgo_));
	  }
	}
      }
    }
  }
}
