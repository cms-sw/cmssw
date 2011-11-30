#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "PhysicsTools/PatExamples/interface/AnalysisTasksAnalyzerJEC.h"
#include "TH2.h"
#include "TProfile.h"
/// default constructor
AnalysisTasksAnalyzerJEC::AnalysisTasksAnalyzerJEC(const edm::ParameterSet& cfg, TFileDirectory& fs): 
  edm::BasicAnalyzer::BasicAnalyzer(cfg, fs),
  Jets_(cfg.getParameter<edm::InputTag>("Jets")),
  jecLevel_(cfg.getParameter<std::string>("jecLevel")),
  patJetCorrFactors_(cfg.getParameter<std::string>("patJetCorrFactors"))
{
 
  hists_["Response"  ] = fs.make<TH2F>("Response"  , "response; #eta;p_{T}(reco)/p_{T}(gen)"  ,  5,  -3.3, 3.3, 100, 0.4, 1.6);
  jetInEvents_=0;
}
AnalysisTasksAnalyzerJEC::~AnalysisTasksAnalyzerJEC()
{

 TFile* file=new TFile("myResponse"+TString(jecLevel_)+".root", "RECREATE");
 TProfile* prof = hists_["Response"  ]->ProfileX();
 prof->Write();
 file->Write();
 file->Close();
}
/// everything that needs to be done during the event loop
void 
AnalysisTasksAnalyzerJEC::analyze(const edm::EventBase& event)
{
  // define what Jet you are using; this is necessary as FWLite is not 
  // capable of reading edm::Views
  using pat::Jet;

  // Handle to the Jet collection
  edm::Handle<std::vector<Jet> > Jets;
  event.getByLabel(Jets_, Jets);

  // loop Jet collection and fill histograms
  for(std::vector<Jet>::const_iterator jet_it=Jets->begin(); jet_it!=Jets->end(); ++jet_it){  

     if(jetInEvents_==0){
       std::cout<<"checking first jet in the events: available JEC levels "<< jet_it->availableJECLevels().size() << std::endl; 
       for(uint k=0; k< jet_it->availableJECLevels().size(); ++k){
	 std::cout<<"available JEC: "<< jet_it->availableJECLevels(patJetCorrFactors_)[k] << std::endl; 
       }
       std::cout<<"available JEC sets "<< jet_it->availableJECSets().size() << std::endl; 
       for(uint k=0; k< jet_it->availableJECSets().size(); ++k){
	 std::cout<<"available JEC: "<< jet_it->availableJECSets()[k] << std::endl; 
       }
     }
     if(jet_it->genParticlesSize()>0){
         hists_["Response" ]->Fill( jet_it->correctedJet(jecLevel_).eta(), jet_it->correctedJet(jecLevel_).pt()/ jet_it->genParticle(0)->pt());
   
     }
     jetInEvents_+=1;
  }
}
