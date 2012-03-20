#include "DQMOffline/PFTau/plugins/PFMETDQMAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/METReco/interface/MET.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//
// -- Constructor
//
PFMETDQMAnalyzer::PFMETDQMAnalyzer(const edm::ParameterSet& parameterSet)  
  
{
  pSet_                = parameterSet;
  inputLabel_          = pSet_.getParameter<edm::InputTag>("InputCollection");
  matchLabel_          = pSet_.getParameter<edm::InputTag>("MatchCollection");
  benchmarkLabel_      = pSet_.getParameter<std::string>("BenchmarkLabel"); 

  pfMETMonitor_.setParameters(parameterSet);  

}
//
// -- BeginJob
//
void PFMETDQMAnalyzer::beginJob() {

  Benchmark::DQM_ = edm::Service<DQMStore>().operator->();
  // part of the following could be put in the base class
  std::string path = "ParticleFlow/" + benchmarkLabel_;
  Benchmark::DQM_->setCurrentFolder(path.c_str());
  edm::LogInfo("PFJMETDQMAnalyzer") << " PFMETDQMAnalyzer::beginJob " <<"Histogram Folder path set to "<< path;
  pfMETMonitor_.setup(pSet_);  
  nBadEvents_ = 0;
}
//
// -- Analyze
//
void PFMETDQMAnalyzer::analyze(edm::Event const& iEvent, 
				      edm::EventSetup const& iSetup) {
  edm::Handle< edm::View<reco::MET> > metCollection;
  iEvent.getByLabel(inputLabel_, metCollection);   
  
  edm::Handle< edm::View<reco::MET> > matchedMetCollection; 
  iEvent.getByLabel( matchLabel_, matchedMetCollection);

  if (metCollection.isValid() && matchedMetCollection.isValid()) {
    float maxRes = 0.0;
    float minRes = 99.99;
    pfMETMonitor_.fillOne( (*metCollection)[0], (*matchedMetCollection)[0], minRes, maxRes);    
    edm::ParameterSet skimPS = pSet_.getParameter<edm::ParameterSet>("SkimParameter");
    if ( (skimPS.getParameter<bool>("switchOn")) && 
         (nBadEvents_ <= skimPS.getParameter<int32_t>("maximumNumberToBeStored")) ) {
      if ( minRes < skimPS.getParameter<double>("lowerCutOffOnResolution")) {
	storeBadEvents(iEvent,minRes);
        nBadEvents_++;
      } else if (maxRes > skimPS.getParameter<double>("upperCutOffOnResolution")) {
        nBadEvents_++;
	storeBadEvents(iEvent,maxRes);
      }
    }
  }
}
void PFMETDQMAnalyzer::storeBadEvents(edm::Event const& iEvent, float& val) {
  unsigned int runNb  = iEvent.id().run();
  unsigned int evtNb  = iEvent.id().event();
  unsigned int lumiNb = iEvent.id().luminosityBlock();
  
  std::string path = "ParticleFlow/" + benchmarkLabel_ + "/BadEvents";
  Benchmark::DQM_->setCurrentFolder(path.c_str());
  std::ostringstream eventid_str;
  eventid_str << runNb << "_"<< evtNb << "_" << lumiNb;
  MonitorElement* me = Benchmark::DQM_->get(path + "/" + eventid_str.str());
  if (me) me->Reset();
  else me = Benchmark::DQM_->bookFloat(eventid_str.str());
  me->Fill(val);  
}

//
// -- EndJob
// 
void PFMETDQMAnalyzer::endJob() {
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE (PFMETDQMAnalyzer) ;
