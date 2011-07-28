#include "DQMOffline/PFTau/plugins/PFCandidateDQMAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//
// -- Constructor
//
PFCandidateDQMAnalyzer::PFCandidateDQMAnalyzer(const edm::ParameterSet& parameterSet)  
  
{
  pSet_                = parameterSet;
  inputLabel_          = pSet_.getParameter<edm::InputTag>("InputCollection");
  matchLabel_          = pSet_.getParameter<edm::InputTag>("MatchCollection");
  benchmarkLabel_      = pSet_.getParameter<std::string>("BenchmarkLabel"); 

  pfCandidateMonitor_.setParameters(parameterSet);  
  
}
//
// -- BeginJob
//
void PFCandidateDQMAnalyzer::beginJob() {

  Benchmark::DQM_ = edm::Service<DQMStore>().operator->();
  // part of the following could be put in the base class
  std::string path = "ParticleFlow/" + benchmarkLabel_;
  Benchmark::DQM_->setCurrentFolder(path.c_str());
  edm::LogInfo("PFCandidateDQMAnalyzer") << " PFCandidateDQMAnalyzer::beginJob " << "Histogram Folder path set to "<< path;
  pfCandidateMonitor_.setup(pSet_);  
  nBadEvents_ = 0;
}
//
// -- Analyze
//
void PFCandidateDQMAnalyzer::analyze(edm::Event const& iEvent, 
				      edm::EventSetup const& iSetup) {
  edm::Handle< edm::View<reco::Candidate> > candCollection;
  iEvent.getByLabel( inputLabel_, candCollection);

  edm::Handle< edm::View<reco::Candidate> > matchedCandCollection;
  iEvent.getByLabel( matchLabel_, matchedCandCollection);

  float maxRes = 0.0;
  float minRes = 99.99;
  if (candCollection.isValid() && matchedCandCollection.isValid()) {
    pfCandidateMonitor_.fill( *candCollection, *matchedCandCollection, minRes, maxRes);
    edm::ParameterSet skimPS = pSet_.getParameter<edm::ParameterSet>("SkimParameter");
    if ( (skimPS.getParameter<bool>("switchOn")) &&  
         (nBadEvents_ <= skimPS.getParameter<int32_t>("maximumNumberToBeStored")) ) {
      if ( minRes < skimPS.getParameter<double>("lowerCutOffOnResolution")) {
	nBadEvents_++; 
	storeBadEvents(iEvent,minRes);
      }	else if (maxRes > skimPS.getParameter<double>("upperCutOffOnResolution")) {
	nBadEvents_++;
	storeBadEvents(iEvent,maxRes);
      }
    }
  }
}
void PFCandidateDQMAnalyzer::storeBadEvents(edm::Event const& iEvent, float& val) {
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
void PFCandidateDQMAnalyzer::endJob() {
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE (PFCandidateDQMAnalyzer) ;
