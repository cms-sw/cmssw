#include "DQMOffline/Trigger/interface/HLTTauDQMOfflineSource.h"
#include "DataFormats/Common/interface/TriggerResults.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace l1extra;
using namespace trigger;

//
// constructors and destructor
//
HLTTauDQMOfflineSource::HLTTauDQMOfflineSource( const edm::ParameterSet& ps ):
  moduleName_(ps.getParameter<std::string>("@module_label")),
  hltProcessName_(ps.getUntrackedParameter<std::string>("HLTProcessName","HLT")),
  triggerResultsSrc_(ps.getUntrackedParameter<edm::InputTag>("TriggerResultsSrc")),
  triggerEventSrc_(ps.getUntrackedParameter<edm::InputTag>("TriggerEventSrc")),
  ps_(ps),
  dqmBaseFolder_(ps.getUntrackedParameter<std::string>("DQMBaseFolder")),
  hltMenuChanged_(true),
  verbose_(ps.getUntrackedParameter<bool>("Verbose",false)),
  automation_(hltProcessName_, L1MatchDr_, HLTMatchDr_),
  L1MatchDr_(ps.getUntrackedParameter<double>("L1MatchDeltaR",0.5)),
  HLTMatchDr_(ps.getUntrackedParameter<double>("HLTMatchDeltaR",0.2)),
  counterEvt_(0)
{}

HLTTauDQMOfflineSource::~HLTTauDQMOfflineSource() {
    //Clear the plotter collections
    while (!l1Plotters.empty()) delete l1Plotters.back(), l1Plotters.pop_back();
    while (!caloPlotters.empty()) delete caloPlotters.back(), caloPlotters.pop_back();
    while (!trackPlotters.empty()) delete trackPlotters.back(), trackPlotters.pop_back();
    while (!pathPlotters.empty()) delete pathPlotters.back(), pathPlotters.pop_back();
    while (!pathPlotters2.empty()) delete pathPlotters2.back(), pathPlotters2.pop_back();
    while (!litePathPlotters.empty()) delete litePathPlotters.back(), litePathPlotters.pop_back();
}

//--------------------------------------------------------
void HLTTauDQMOfflineSource::beginJob() {    
}

//--------------------------------------------------------
void HLTTauDQMOfflineSource::beginRun( const edm::Run& iRun, const EventSetup& iSetup ) {
    //Evaluate configuration for every new trigger menu
    if ( HLTCP_.init(iRun, iSetup, hltProcessName_, hltMenuChanged_) ) {
        if ( hltMenuChanged_ ) {
            processPSet(ps_);
            if (verbose_) {
                std::cout << "Trigger menu '" << HLTCP_.tableName() << "'" << std::endl;
                HLTCP_.dump("Triggers");
                
                std::cout << std::endl << "Configuration of '" << moduleName_ << "' for trigger menu '" << HLTCP_.tableName() << "'" << std::endl;
                for ( unsigned int i = 0; i < config_.size(); ++i ) {
                    std::cout << config_[i].dump() << std::endl;
                }
                std::cout << matching_.dump() << std::endl << std::endl;
                
                unsigned int npars = 14;
                npars += countParameters(matching_);
                for ( unsigned int i = 0; i < config_.size(); ++i ) {
                    npars += countParameters(config_[i]);
                }
                
                std::cout << "--> Number of parameters: " << npars << std::endl;
                std::cout << std::endl << "Event content need by this module: " << std::endl;
                
                std::vector<edm::InputTag> evtcontent;
                for ( unsigned int i = 0; i < config_.size(); ++i ) {
                    searchEventContent(evtcontent, config_[i]);
                }
                searchEventContent(evtcontent, matching_);
                
                for (std::vector<edm::InputTag>::const_iterator iter = evtcontent.begin(); iter != evtcontent.end(); ++iter) {
                    std::cout << " " << iter->encode() << std::endl;
                }
            }
        }
    } else {
        edm::LogWarning("HLTTauDQMOfflineSource") << "HLT config extraction failure with process name '" << hltProcessName_ << "'";
    }
}

//--------------------------------------------------------
void HLTTauDQMOfflineSource::beginLuminosityBlock( const LuminosityBlock& lumiSeg, const EventSetup& context ) {
}

// ----------------------------------------------------------
void HLTTauDQMOfflineSource::analyze(const Event& iEvent, const EventSetup& iSetup ) {
    //Apply the prescaler
    if (counterEvt_ > prescaleEvt_) {
        //Do Analysis here
        counterEvt_ = 0;

        edm::Handle<edm::TriggerResults> triggerResultsHandle;
        iEvent.getByLabel(triggerResultsSrc_, triggerResultsHandle);
        if(!triggerResultsHandle.isValid()) {
          edm::LogWarning("HLTTauDQMOffline") << "Unable to read edm::TriggerResults with label " << triggerResultsSrc_;
        }

        edm::Handle<trigger::TriggerEvent> triggerEventHandle;
        iEvent.getByLabel(triggerEventSrc_, triggerEventHandle);
        if(!triggerEventHandle.isValid()) {
          edm::LogWarning("HLTTauDQMOffline") << "Unable to read trigger::TriggerEvent with label " << triggerEventSrc_;
          return;
        }


        //Create match collections
        std::map<int,LVColl> refC;
        
        if (doRefAnalysis_) {
            for ( std::vector<edm::ParameterSet>::const_iterator iter = refObjects_.begin(); iter != refObjects_.end(); ++iter ) {
                int objID = iter->getUntrackedParameter<int>("matchObjectID");
                
                Handle<LVColl> collHandle;
                if ( iEvent.getByLabel(iter->getUntrackedParameter<edm::InputTag>("FilterName"),collHandle) ) {
                    std::map<int,LVColl>::iterator it;

                    it = refC.find(objID);
                    if ( it == refC.end() ) {
                      refC.insert(std::pair<int,LVColl>(objID,*collHandle));
                    } else {
                      it->second.insert(it->second.end(),collHandle->begin(),collHandle->end());
                    }
                }
            }
        }
        
        //Path Plotters
        for ( unsigned int i = 0; i < pathPlotters.size(); ++i ) {
            if (pathPlotters[i]->isValid()) pathPlotters[i]->analyze(iEvent,iSetup,refC);
        }
        for ( unsigned int i = 0; i < pathPlotters2.size(); ++i ) {
          if (pathPlotters2[i]->isValid()) pathPlotters2[i]->analyze(*triggerResultsHandle, *triggerEventHandle, refC);
        }
        
        //Lite Path Plotters
        for ( unsigned int i = 0; i < litePathPlotters.size(); ++i ) {
            if (litePathPlotters[i]->isValid()) litePathPlotters[i]->analyze(iEvent,iSetup,refC);
        }
        
        //L1 Plotters
        for ( unsigned int i = 0; i < l1Plotters.size(); ++i ) {
            if (l1Plotters[i]->isValid()) l1Plotters[i]->analyze(iEvent,iSetup,refC);
        }
        
        //Calo Plotters
        for ( unsigned int i = 0; i < caloPlotters.size(); ++i ) {
            if (caloPlotters[i]->isValid()) caloPlotters[i]->analyze(iEvent,iSetup,refC);            
        }
        
        //Track Plotters
        for ( unsigned int i = 0; i < trackPlotters.size(); ++i ) {
            if (trackPlotters[i]->isValid()) trackPlotters[i]->analyze(iEvent,iSetup,refC);
        }
    } else {
        counterEvt_++;
    }
}

//--------------------------------------------------------
void HLTTauDQMOfflineSource::endLuminosityBlock( const LuminosityBlock& lumiSeg, const EventSetup& context ) {
}

//--------------------------------------------------------
void HLTTauDQMOfflineSource::endRun( const Run& r, const EventSetup& context ) {
}

//--------------------------------------------------------
void HLTTauDQMOfflineSource::endJob() {
    return;
}

void HLTTauDQMOfflineSource::processPSet( const edm::ParameterSet& pset ) {
    //Get General Monitoring Parameters
    config_        = pset.getParameter<std::vector<edm::ParameterSet> >("MonitorSetup");
    matching_      = pset.getParameter<edm::ParameterSet>("Matching");
    NPtBins_       = pset.getUntrackedParameter<int>("PtHistoBins",20);
    NEtaBins_      = pset.getUntrackedParameter<int>("EtaHistoBins",25);
    NPhiBins_      = pset.getUntrackedParameter<int>("PhiHistoBins",32);
    EtMax_         = pset.getUntrackedParameter<double>("EtHistoMax",100);
    prescaleEvt_   = pset.getUntrackedParameter<int>("prescaleEvt", -1);
    doRefAnalysis_ = matching_.getUntrackedParameter<bool>("doMatching");
    refObjects_    = matching_.getUntrackedParameter<std::vector<edm::ParameterSet> >("matchFilters");
    
    //Clear the plotter collections first
    while (!l1Plotters.empty()) delete l1Plotters.back(), l1Plotters.pop_back();
    while (!caloPlotters.empty()) delete caloPlotters.back(), caloPlotters.pop_back();
    while (!trackPlotters.empty()) delete trackPlotters.back(), trackPlotters.pop_back();
    while (!pathPlotters.empty()) delete pathPlotters.back(), pathPlotters.pop_back();
    while (!pathPlotters2.empty()) delete pathPlotters2.back(), pathPlotters2.pop_back();
    while (!litePathPlotters.empty()) delete litePathPlotters.back(), litePathPlotters.pop_back();
    
    //Automatic Configuration
    automation_.AutoCompleteConfig( config_, HLTCP_ );

    //Read The Configuration
    for ( unsigned int i = 0; i < config_.size(); ++i ) {
        std::string configtype;
        try {
            configtype = config_[i].getUntrackedParameter<std::string>("ConfigType");
        } catch ( cms::Exception &e ) {
            edm::LogWarning("HLTTauDQMOfflineSource")
            << e.what() << std::endl;
            continue;
        }
        if (configtype == "L1") {
            try {
                l1Plotters.push_back(new HLTTauDQML1Plotter(config_[i],NPtBins_,NEtaBins_,NPhiBins_,EtMax_,doRefAnalysis_,L1MatchDr_,dqmBaseFolder_));
            } catch ( cms::Exception &e ) {
                edm::LogWarning("HLTTauDQMSource") << e.what() << std::endl;
                continue;
            }
        } else if (configtype == "Calo") {
            try {
                caloPlotters.push_back(new HLTTauDQMCaloPlotter(config_[i],NPtBins_,NEtaBins_,NPhiBins_,EtMax_,doRefAnalysis_,HLTMatchDr_,dqmBaseFolder_));
            } catch ( cms::Exception &e ) {
                edm::LogWarning("HLTTauDQMSource") << e.what() << std::endl;
                continue;
            }
        } else if (configtype == "Track") {
            try {
                trackPlotters.push_back(new HLTTauDQMTrkPlotter(config_[i],NPtBins_,NEtaBins_,NPhiBins_,EtMax_,doRefAnalysis_,HLTMatchDr_,dqmBaseFolder_));    
            } catch ( cms::Exception &e ) {
                edm::LogWarning("HLTTauDQMSource") << e.what() << std::endl;
                continue;
            }
        } else if (configtype == "Path") {
            try {
                pathPlotters.push_back(new HLTTauDQMPathPlotter(config_[i],doRefAnalysis_,dqmBaseFolder_));
            } catch ( cms::Exception &e ) {
                edm::LogWarning("HLTTauDQMSource") << e.what() << std::endl;
                continue;
            }
        } else if (configtype == "Path2") {
            try {
              pathPlotters2.push_back(new HLTTauDQMPathPlotter2(config_[i], doRefAnalysis_, dqmBaseFolder_, HLTCP_));
            } catch ( cms::Exception &e ) {
              edm::LogWarning("HLTTauDQMSource") << e.what() << std::endl;
              continue;
            }
        } else if (configtype == "LitePath") {
            try {
                litePathPlotters.push_back(new HLTTauDQMLitePathPlotter(config_[i],NPtBins_,NEtaBins_,NPhiBins_,EtMax_,doRefAnalysis_,HLTMatchDr_,dqmBaseFolder_));   
            } catch ( cms::Exception &e ) {
                edm::LogWarning("HLTTauDQMSource") << e.what() << std::endl;
                continue;
            }
        }
    }
}

unsigned int HLTTauDQMOfflineSource::countParameters( const edm::ParameterSet& pset ) {
    unsigned int num = 0;
    const std::map<std::string,edm::ParameterSetEntry>& tmppset = pset.psetTable();
    for ( std::map<std::string,edm::ParameterSetEntry>::const_iterator iter = tmppset.begin(); iter != tmppset.end(); ++iter ) {
        num += countParameters(iter->second.pset());
    }
    const std::map<std::string,edm::VParameterSetEntry>& tmpvpset = pset.vpsetTable();
    for ( std::map<std::string,edm::VParameterSetEntry>::const_iterator iter = tmpvpset.begin(); iter != tmpvpset.end(); ++iter ) {
        const std::vector<edm::ParameterSet>& tmpvec = iter->second.vpset();
        for ( std::vector<edm::ParameterSet>::const_iterator iter2 = tmpvec.begin(); iter2 != tmpvec.end(); ++iter2 ) {
            num += countParameters(*iter2);
        }
    }
    num += pset.tbl().size();
    return num;
}

void HLTTauDQMOfflineSource::searchEventContent(std::vector<edm::InputTag>& eventContent, const edm::ParameterSet& pset) {
    for (std::map< std::string, edm::Entry >::const_iterator i = pset.tbl().begin(), e = pset.tbl().end(); i != e; ++i) {
        if (std::string(1,i->second.typeCode()) == "t") {
            std::vector<edm::InputTag>::iterator iter = std::find(eventContent.begin(), eventContent.end(), i->second.getInputTag());
            if (iter == eventContent.end()) {
                eventContent.push_back(i->second.getInputTag());
            }
        }
    }
    for (std::map< std::string, edm::ParameterSetEntry >::const_iterator i = pset.psetTable().begin(), e = pset.psetTable().end(); i != e; ++i) {
        searchEventContent(eventContent, i->second.pset());
    }
    for (std::map< std::string, edm::VParameterSetEntry >::const_iterator i = pset.vpsetTable().begin(), e = pset.vpsetTable().end(); i != e; ++i) {
        std::vector<edm::ParameterSet> vpset = i->second.vpset();
        for (std::vector<edm::ParameterSet>::const_iterator iter = vpset.begin(); iter != vpset.end(); ++iter) {
            searchEventContent(eventContent, *iter);
        }
    }
}
