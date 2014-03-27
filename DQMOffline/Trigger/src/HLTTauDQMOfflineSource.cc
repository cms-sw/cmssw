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
  counterEvt_(0),
  prescaleEvt_(ps.getUntrackedParameter<int>("prescaleEvt", -1))
{
  int nPtBins  = ps.getUntrackedParameter<int>("PtHistoBins", 20);
  int nEtaBins = ps.getUntrackedParameter<int>("EtaHistoBins",12);
  int nPhiBins = ps.getUntrackedParameter<int>("PhiHistoBins",18);
  double ptMax = ps.getUntrackedParameter<double>("PtHistoMax",200);
  double highPtMax = ps.getUntrackedParameter<double>("HighPtHistoMax",1000);
  double l1MatchDr = ps.getUntrackedParameter<double>("L1MatchDeltaR", 0.5);
  double hltMatchDr = ps.getUntrackedParameter<double>("HLTMatchDeltaR", 0.2);
  std::string dqmBaseFolder = ps.getUntrackedParameter<std::string>("DQMBaseFolder");

  edm::ParameterSet matching = ps.getParameter<edm::ParameterSet>("Matching");
  doRefAnalysis_ = matching.getUntrackedParameter<bool>("doMatching");

  using VPSet = std::vector<edm::ParameterSet>;
  VPSet monitorSetup = ps.getParameter<VPSet>("MonitorSetup");

  /*
  l1Plotters_.reserve(monitorSetup.size());
  pathPlotters_.reserve(monitorSetup.size());
  pathSummaryPlotters_.reserve(monitorSetup.size());
  */
  for(const edm::ParameterSet& pset: monitorSetup) {
    std::string configtype;
    try {
      configtype = pset.getUntrackedParameter<std::string>("ConfigType");
    } catch(cms::Exception& e) {
      edm::LogWarning("HLTTauDQMOffline") << e.what() << std::endl;
      continue;
    }
    if(configtype == "L1") {
      try {
        l1Plotters_.emplace_back(pset, consumesCollector(), nPhiBins, ptMax, highPtMax, doRefAnalysis_, l1MatchDr, dqmBaseFolder);
      } catch(cms::Exception& e) {
        edm::LogWarning("HLTTauDQMOffline") << e.what() << std::endl;
        continue;
      }
    } else if (configtype == "Path") {
      try {
        pathPlotters2_.emplace_back(pset, doRefAnalysis_, dqmBaseFolder, hltProcessName_, nPtBins, nEtaBins, nPhiBins, ptMax, highPtMax, l1MatchDr, hltMatchDr);
      } catch ( cms::Exception &e ) {
        edm::LogWarning("HLTTauDQMOffline") << e.what() << std::endl;
        continue;
      }
    } else if (configtype == "PathSummary") {
      try {
        pathSummaryPlotters_.emplace_back(pset, doRefAnalysis_, dqmBaseFolder, hltMatchDr);
      } catch ( cms::Exception &e ) {
        edm::LogWarning("HLTTauDQMOffline") << e.what() << std::endl;
        continue;
      }
    }
  }

  if(doRefAnalysis_) {
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
    if(hltMenuChanged) {
      std::vector<const HLTTauDQMPath *> pathObjects;
      pathObjects.reserve(pathPlotters2_.size());
      for(auto& pathPlotter: pathPlotters2_) {
        pathPlotter.updateHLTMenu(HLTCP_);
        if(pathPlotter.isValid())
          pathObjects.push_back(pathPlotter.getPathObject());
      }
      for(auto& pathSummaryPlotter: pathSummaryPlotters_) {
        pathSummaryPlotter.setPathObjects(pathObjects);
      }
    }
  } else {
    edm::LogWarning("HLTTauDQMOffline") << "HLT config extraction failure with process name '" << hltProcessName_ << "'";
  }
}

//--------------------------------------------------------
void HLTTauDQMOfflineSource::bookHistograms(DQMStore::IBooker &iBooker, const edm::Run& iRun, const EventSetup& iSetup ) {
  for(auto& l1Plotter: l1Plotters_) {
    l1Plotter.bookHistograms(iBooker);
  }
  for(auto& pathPlotter: pathPlotters2_) {
    pathPlotter.bookHistograms(iBooker);
  }
  for(auto& pathSummaryPlotter: pathSummaryPlotters_) {
    pathSummaryPlotter.bookHistograms(iBooker);
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
          }
        }
        
        //Path Plotters
        for(auto& pathPlotter: pathPlotters2_) {
          if(pathPlotter.isValid())
            pathPlotter.analyze(*triggerResultsHandle, *triggerEventHandle, refC);
        }
        
        for(auto& pathSummaryPlotter: pathSummaryPlotters_) {
          if(pathSummaryPlotter.isValid())
            pathSummaryPlotter.analyze(*triggerResultsHandle, *triggerEventHandle, refC);
        }
        
        //L1 Plotters
        for(auto& l1Plotter: l1Plotters_) {
          if(l1Plotter.isValid())
            l1Plotter.analyze(iEvent, iSetup, refC);
        }
    } else {
        counterEvt_++;
    }
}
