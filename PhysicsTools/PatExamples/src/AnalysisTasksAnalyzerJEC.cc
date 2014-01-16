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
  jecSetLabel_(cfg.getParameter<std::string>("jecSetLabel"))
{
  hists_["Response"  ] = fs.make<TH2F>("Response" , "response "+TString(jecLevel_)+"; #eta;p_{T}(reco)/p_{T}(gen)"  ,  5,  -3.3, 3.3, 100, 0.4, 1.6);
}
/// deconstructor
AnalysisTasksAnalyzerJEC::~AnalysisTasksAnalyzerJEC()
{
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


///// You can switch message logger messages on in the configuration file to get some help.
      for(unsigned int k=0; k< jet_it->availableJECSets().size(); ++k){
	  edm::LogInfo ("hint1") <<"\n \n *************** HINT 1 *************** \n \n Label of available JEC Set: "<< jet_it->availableJECSets()[k] 
				 <<"\n \n You found out which JEC Sets were created within the PAT tuple creation! The wrong label caused the segmentation fault in the next for-loop where you ask for JEC levels of a JEC Set that does not exist. Correct the JEC label in your config file and eliminate the segmentation fault. \n  *********************************************** \n"; 
      }	     		
      for(unsigned int k=0; k< jet_it->availableJECLevels().size(); ++k){
	  edm::LogInfo("hint2")<<" \n \n  Label of available JEC level: "<< jet_it->availableJECLevels(jecSetLabel_)[k] << "\n \n "; 
      }
      edm::LogInfo("hint2")<<"\n \n  *************** HINT 2 ************** \n You did it correctly congratulations!!!!  And you found out above which JEC levels are saved within the jets. We want to investigate these in the following with the response function. At the moment you are trying to access the JEC Level: "
			   << jecLevel_ << ". Does it exist? This causes the error below (see 'Exception Message')! Correct the label of the correction level to an existing one that you are interested in. \n ******************************************* \n " <<std::endl;	     		
      if(jet_it->genParticlesSize()>0){
	  hists_["Response" ]->Fill( jet_it->correctedJet(jecLevel_).eta(), jet_it->correctedJet(jecLevel_).pt()/ jet_it->genParticle(0)->pt());
      }
  }
}
