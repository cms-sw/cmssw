#include "DQMOffline/PFTau/plugins/PFJetDQMAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//
// -- Constructor
//
PFJetDQMAnalyzer::PFJetDQMAnalyzer(const edm::ParameterSet& parameterSet)  
  
{
  pSet_                = parameterSet;
  inputLabel_          = pSet_.getParameter<edm::InputTag>("InputCollection");
  matchLabel_          = pSet_.getParameter<edm::InputTag>("MatchCollection");
  benchmarkLabel_      = pSet_.getParameter<std::string>("BenchmarkLabel"); 

  pfJetMonitor_.setParameters(parameterSet);  // set parameters for booking histograms and validating jet 
  
  myJet_ = consumes< edm::View<reco::Jet> >(inputLabel_);
  myMatchedJet_ = consumes< edm::View<reco::Jet> >(matchLabel_);


  std::string folder = benchmarkLabel_ ;

  subsystemname_ = "ParticleFlow" ;
  eventInfoFolder_ = subsystemname_ + "/" + folder ;

  nBadEvents_ = 0;

}


//
// -- BookHistograms
//
void PFJetDQMAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
					    edm::Run const & /* iRun */,
					    edm::EventSetup const & /* iSetup */ )
{
  ibooker.setCurrentFolder(eventInfoFolder_) ;

  edm::LogInfo("PFJetDQMAnalyzer") << " PFJetDQMAnalyzer::beginJob " << "Histogram Folder path set to " << eventInfoFolder_;

  pfJetMonitor_.setup(ibooker, pSet_);
}


//
// -- Analyze
//
void PFJetDQMAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {

  edm::Handle< edm::View<reco::Jet> > jetCollection;
  iEvent.getByToken(myJet_, jetCollection);   
  
  edm::Handle< edm::View<reco::Jet> > matchedJetCollection; 
  iEvent.getByToken(myMatchedJet_, matchedJetCollection);

  float maxRes = 0.0;
  float minRes = 99.99;
  float jetpT = 0.0;
  if (jetCollection.isValid() && matchedJetCollection.isValid()) {
    pfJetMonitor_.fill( *jetCollection, *matchedJetCollection, minRes, maxRes, jetpT, pSet_);  // match collections and fill pt eta phi and charge histos for candidate jet, fill delta_x_VS_y histos for matched couples, book and fill delta_frac_VS_frac histos for matched couples

    /*
    edm::ParameterSet skimPS = pSet_.getParameter<edm::ParameterSet>("SkimParameter");
    if ( (skimPS.getParameter<bool>("switchOn")) &&  
         (nBadEvents_ <= skimPS.getParameter<int32_t>("maximumNumberToBeStored")) ) {
      if (jetpT > skimPS.getParameter<double>("minimumJetpT")) { 
	if ( minRes < skimPS.getParameter<double>("lowerCutOffOnResolution")) {
	  //storeBadEvents(iEvent,minRes);
	  storeBadEvents(ibooker, iEvent,minRes);
	  nBadEvents_++;
	} else if (maxRes > skimPS.getParameter<double>("upperCutOffOnResolution")) {
	  //storeBadEvents(iEvent,maxRes);
	  storeBadEvents(ibooker, iEvent,maxRes);
	  nBadEvents_++;
	}
      } // minimum jet pT check
    }
    */

  }
}


void PFJetDQMAnalyzer::storeBadEvents(DQMStore::IBooker & ibooker, edm::Event const& iEvent, float& val) {
  unsigned int runNb  = iEvent.id().run();
  unsigned int evtNb  = iEvent.id().event();
  unsigned int lumiNb = iEvent.id().luminosityBlock();
  
  std::string path = "ParticleFlow/" + benchmarkLabel_ + "/BadEvents";
  ibooker.setCurrentFolder(eventInfoFolder_) ;
  std::ostringstream eventid_str;
  eventid_str << runNb << "_"<< evtNb << "_" << lumiNb;

  /*
  MonitorElement* me = ibooker.get(path + "/" + eventid_str.str());
  if (me) me->Reset();
  else {
    me = ibooker.bookFloat(eventid_str.str());
  }
  me->Fill(val);  
  */
  MonitorElement* me = ibooker.bookFloat(eventid_str.str());
  me->Fill(val);  

}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE (PFJetDQMAnalyzer) ;
