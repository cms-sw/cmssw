#include "DQMOffline/Trigger/interface/HLTTauDQMOfflineSource.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace l1extra;
using namespace trigger;

//
// constructors and destructor
//
HLTTauDQMOfflineSource::HLTTauDQMOfflineSource( const edm::ParameterSet& ps ):
  hltProcessName_(ps.getUntrackedParameter<std::string>("HLTProcessName","HLT")),
  triggerResultsSrc_(ps.getUntrackedParameter<edm::InputTag>("TriggerResultsSrc")),
  triggerResultsToken_(consumes<edm::TriggerResults>(triggerResultsSrc_)),
  triggerEventSrc_(ps.getUntrackedParameter<edm::InputTag>("TriggerEventSrc")),
  triggerEventToken_(consumes<trigger::TriggerEvent>(triggerEventSrc_)),
  pathRegex_(ps.getUntrackedParameter<std::string>("Paths")),
  nPtBins_(ps.getUntrackedParameter<int>("PtHistoBins", 20)),
  nEtaBins_(ps.getUntrackedParameter<int>("EtaHistoBins",12)),
  nPhiBins_(ps.getUntrackedParameter<int>("PhiHistoBins",18)),
  ptMax_(ps.getUntrackedParameter<double>("PtHistoMax",200)),
  highPtMax_(ps.getUntrackedParameter<double>("HighPtHistoMax",1000)),
  l1MatchDr_(ps.getUntrackedParameter<double>("L1MatchDeltaR", 0.5)),
  hltMatchDr_(ps.getUntrackedParameter<double>("HLTMatchDeltaR", 0.5)),
  dqmBaseFolder_(ps.getUntrackedParameter<std::string>("DQMBaseFolder")),
  counterEvt_(0),
  prescaleEvt_(ps.getUntrackedParameter<int>("prescaleEvt", -1))
{
  edm::ParameterSet matching = ps.getParameter<edm::ParameterSet>("Matching");
  doRefAnalysis_ = matching.getUntrackedParameter<bool>("doMatching");

  if(ps.exists("L1Plotter")) {
    l1Plotter_.reset(new HLTTauDQML1Plotter(ps.getUntrackedParameter<edm::ParameterSet>("L1Plotter"), consumesCollector(),
                                            nPhiBins_, ptMax_, highPtMax_, doRefAnalysis_, l1MatchDr_, dqmBaseFolder_));
  }
  if(ps.exists("PathSummaryPlotter")) {
    pathSummaryPlotter_.reset(new HLTTauDQMPathSummaryPlotter(ps.getUntrackedParameter<edm::ParameterSet>("PathSummaryPlotter"),
                                                              doRefAnalysis_, dqmBaseFolder_, hltMatchDr_));
  }

  if(doRefAnalysis_) {
    using VPSet = std::vector<edm::ParameterSet>;
    VPSet matchObjects = matching.getUntrackedParameter<VPSet>("matchFilters");
    for(const edm::ParameterSet& pset: matchObjects) {
      refObjects_.push_back(RefObject{pset.getUntrackedParameter<int>("matchObjectID"),
            consumes<LVColl>(pset.getUntrackedParameter<edm::InputTag>("FilterName"))});
    }
  }
}

HLTTauDQMOfflineSource::~HLTTauDQMOfflineSource() {
}

//--------------------------------------------------------
void HLTTauDQMOfflineSource::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  //Evaluate configuration for every new trigger menu
  bool hltMenuChanged = false;
  if(HLTCP_.init(iRun, iSetup, hltProcessName_, hltMenuChanged)) {
    LogDebug("HLTTauDQMOffline") << "dqmBeginRun(), hltMenuChanged " << hltMenuChanged;
    if(hltMenuChanged) {
      // Find all paths to monitor
      std::vector<std::string> foundPaths;
      boost::smatch what;
      LogDebug("HLTTauDQMOffline") << "Looking for paths with regex " << pathRegex_;
      for(const std::string& pathName: HLTCP_.triggerNames()) {
        if(boost::regex_search(pathName, what, pathRegex_)) {
          LogDebug("HLTTauDQMOffline") << "Found path " << pathName;
          foundPaths.emplace_back(pathName);
        }
      }
      std::sort(foundPaths.begin(), foundPaths.end());

      // Construct path plotters
      std::vector<const HLTTauDQMPath *> pathObjects;
      pathPlotters_.reserve(foundPaths.size());
      pathObjects.reserve(foundPaths.size());
      for(const std::string& pathName: foundPaths) {
        pathPlotters_.emplace_back(pathName, HLTCP_, doRefAnalysis_, dqmBaseFolder_, hltProcessName_, nPtBins_, nEtaBins_, nPhiBins_, ptMax_, highPtMax_, l1MatchDr_, hltMatchDr_);
        if(pathPlotters_.back().isValid()) {
          pathObjects.push_back(pathPlotters_.back().getPathObject());
        }
      }

      // Update paths to the summary plotter
      if(pathSummaryPlotter_) {
        pathSummaryPlotter_->setPathObjects(pathObjects);
      }
    }
  } else {
    edm::LogWarning("HLTTauDQMOffline") << "HLT config extraction failure with process name '" << hltProcessName_ << "'";
  }
}

//--------------------------------------------------------
void HLTTauDQMOfflineSource::bookHistograms(DQMStore::IBooker &iBooker, const edm::Run& iRun, const EventSetup& iSetup ) {
  if(l1Plotter_) {
    l1Plotter_->bookHistograms(iBooker);
  }
  for(auto& pathPlotter: pathPlotters_) {
    pathPlotter.bookHistograms(iBooker);
  }
  if(pathSummaryPlotter_) {
    pathSummaryPlotter_->bookHistograms(iBooker);
  }
}

// ----------------------------------------------------------
void HLTTauDQMOfflineSource::analyze(const Event& iEvent, const EventSetup& iSetup ) {
    //Apply the prescaler
    if (counterEvt_ > prescaleEvt_) {
        //Do Analysis here
        counterEvt_ = 0;

        edm::Handle<edm::TriggerResults> triggerResultsHandle;
        iEvent.getByToken(triggerResultsToken_, triggerResultsHandle);
        if(!triggerResultsHandle.isValid()) {
          edm::LogWarning("HLTTauDQMOffline") << "Unable to read edm::TriggerResults with label " << triggerResultsSrc_;
          return;
        }

        edm::Handle<trigger::TriggerEvent> triggerEventHandle;
        iEvent.getByToken(triggerEventToken_, triggerEventHandle);
        if(!triggerEventHandle.isValid()) {
          edm::LogWarning("HLTTauDQMOffline") << "Unable to read trigger::TriggerEvent with label " << triggerEventSrc_;
          return;
        }

        //Create match collections
        HLTTauDQMOfflineObjects refC;
        if (doRefAnalysis_) {
          for(RefObject& refObj: refObjects_) {
            edm::Handle<LVColl> collHandle;
            iEvent.getByToken(refObj.token, collHandle);
            if(!collHandle.isValid())
              continue;

            if(refObj.objID == 11) {
              refC.electrons.insert(refC.electrons.end(), collHandle->begin(), collHandle->end());
            }
            else if(refObj.objID == 13) {
              refC.muons.insert(refC.muons.end(), collHandle->begin(), collHandle->end());
            }
            else if(refObj.objID == 15) {
              refC.taus.insert(refC.taus.end(), collHandle->begin(), collHandle->end());
            }
            else if(refObj.objID == 0) {
              refC.met.insert(refC.met.end(), collHandle->begin(), collHandle->end());
            }
          }
        }
        
        //Path Plotters
        for(auto& pathPlotter: pathPlotters_) {
          if(pathPlotter.isValid())
            pathPlotter.analyze(*triggerResultsHandle, *triggerEventHandle, refC);
        }
        
        if(pathSummaryPlotter_ && pathSummaryPlotter_->isValid()) {
          pathSummaryPlotter_->analyze(*triggerResultsHandle, *triggerEventHandle, refC);
        }
        
        //L1 Plotter
        if(l1Plotter_ && l1Plotter_->isValid()) {
          l1Plotter_->analyze(iEvent, iSetup, refC);
        }
    } else {
        counterEvt_++;
    }
}
