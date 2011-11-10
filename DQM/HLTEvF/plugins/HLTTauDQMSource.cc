#include "DQM/HLTEvF/interface/HLTTauDQMSource.h"

//
// constructors and destructor
//
HLTTauDQMSource::HLTTauDQMSource( const edm::ParameterSet& ps ) {
    //Get initialization
    moduleName_     = ps.getUntrackedParameter<std::string>("ModuleName");
    dqmBaseFolder_  = ps.getUntrackedParameter<std::string>("DQMBaseFolder");
    hltProcessName_ = ps.getUntrackedParameter<std::string>("HLTProcessName","HLT");
    L1MatchDr_      = ps.getUntrackedParameter<double>("L1MatchDeltaR",0.5);
    HLTMatchDr_     = ps.getUntrackedParameter<double>("HLTMatchDeltaR",0.2);
    verbose_        = ps.getUntrackedParameter<bool>("Verbose",false);
    counterEvt_     = 0;
    hltMenuChanged_ = true;
    automation_     = HLTTauDQMAutomation(hltProcessName_, L1MatchDr_, HLTMatchDr_);     
    ps_             = ps;
}

HLTTauDQMSource::~HLTTauDQMSource() {
    //Clear the plotter collections
    while (!l1Plotters.empty()) delete l1Plotters.back(), l1Plotters.pop_back();
    while (!caloPlotters.empty()) delete caloPlotters.back(), caloPlotters.pop_back();
    while (!trackPlotters.empty()) delete trackPlotters.back(), trackPlotters.pop_back();
    while (!pathPlotters.empty()) delete pathPlotters.back(), pathPlotters.pop_back();
    while (!litePathPlotters.empty()) delete litePathPlotters.back(), litePathPlotters.pop_back();
}

//--------------------------------------------------------
void HLTTauDQMSource::beginJob() {
}

//--------------------------------------------------------
void HLTTauDQMSource::beginRun( const edm::Run& iRun, const edm::EventSetup& iSetup ) {
    //Evaluate configuration for every new trigger menu
    if ( HLTCP_.init(iRun, iSetup, hltProcessName_, hltMenuChanged_) ) {
        if ( hltMenuChanged_ ) {
            processPSet(ps_);
            if (verbose_) {
                std::cout << "Configuration of '" << moduleName_ << "' for trigger menu '" << HLTCP_.tableName() << "'" << std::endl;
                for ( unsigned int i = 0; i < config_.size(); ++i ) {
                    std::cout << config_[i].dump() << std::endl;
                }
                std::cout << matching_.dump() << std::endl << std::endl;
            }
        }
    } else {
        edm::LogWarning("HLTTauDQMSource") << "HLT config extraction failure with process name '" << hltProcessName_ << "'";
    }
}

//--------------------------------------------------------
void HLTTauDQMSource::beginLuminosityBlock( const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context ) {
}

// ----------------------------------------------------------
void HLTTauDQMSource::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {
    using namespace std;
    using namespace edm;
    using namespace reco;
    using namespace l1extra;
    using namespace trigger;
    
    //Apply the prescaler
    if (counterEvt_ > prescaleEvt_) {
        //Do analysis here
        counterEvt_ = 0;
        
        //Create match collections
        std::map<int,LVColl> refC;
        
        if (doRefAnalysis_) {
            Handle<TriggerEvent> trigEv;
            
            //Get the TriggerEvent
            bool gotTEV = true;
            try {
                gotTEV = iEvent.getByLabel(refTriggerEvent_,trigEv);
            } catch (cms::Exception& exception) {
                gotTEV = false;
            }
            
            if (gotTEV) {
                for ( std::vector<edm::ParameterSet>::const_iterator iter = refFilters_.begin(); iter != refFilters_.end(); ++iter ) {
                    size_t ID = trigEv->filterIndex(iter->getUntrackedParameter<edm::InputTag>("FilterName"));
                    int objID = iter->getUntrackedParameter<int>("matchObjectID");
                    double objMinPt = iter->getUntrackedParameter<double>("matchObjectMinPt");
                    
                    std::map<int,LVColl>::iterator it;
                    LVColl tmpColl = getFilterCollection(ID,objID,*trigEv,objMinPt);
                    
                    //Set all filter collections to be 'tau like' until we have a better solution for reference objects
                    it = refC.find(15);
                    if ( it == refC.end() ) {
                        refC.insert(std::pair<int,LVColl>(15,tmpColl));
                    } else {
                        it->second.insert(it->second.end(),tmpColl.begin(),tmpColl.end());
                    }
                    
//                    if ( objID == trigger::TriggerL1TauJet || objID == trigger::TriggerTau || std::abs(objID) == 15 ) {
//                        it = refC.find(15);
//                        if ( it == refC.end() ) {
//                            refC.insert(std::pair<int,LVColl>(15,tmpColl));
//                        } else {
//                            it->second.insert(it->second.end(),tmpColl.begin(),tmpColl.end());
//                        }
//                    } else if ( objID == trigger::TriggerL1IsoEG || objID == trigger::TriggerL1NoIsoEG || objID == trigger::TriggerElectron || std::abs(objID) == 11 ) {
//                        it = refC.find(11);
//                        if ( it == refC.end() ) {
//                            refC.insert(std::pair<int,LVColl>(11,tmpColl));
//                        } else {
//                            it->second.insert(it->second.end(),tmpColl.begin(),tmpColl.end());
//                        }
//                    } else if ( objID == trigger::TriggerL1Mu || objID == trigger::TriggerMuon || std::abs(objID) == 13 ) {
//                        it = refC.find(13);
//                        if ( it == refC.end() ) {
//                            refC.insert(std::pair<int,LVColl>(13,tmpColl));
//                        } else {
//                            it->second.insert(it->second.end(),tmpColl.begin(),tmpColl.end());
//                        }
//                    } else {
//                        it = refC.find(objID);
//                        if ( it == refC.end() ) {
//                            refC.insert(std::pair<int,LVColl>(objID,tmpColl));
//                        } else {
//                            it->second.insert(it->second.end(),tmpColl.begin(),tmpColl.end());
//                        }
//                    }
                }
            }
        }
        
        //Path Plotters
        for ( unsigned int i = 0; i < pathPlotters.size(); ++i ) {
            if (pathPlotters[i]->isValid()) pathPlotters[i]->analyze(iEvent,iSetup,refC);
        }
        
        //Lite Path Plotters
        for (unsigned int i=0;i<litePathPlotters.size();++i) {
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
void HLTTauDQMSource::endLuminosityBlock( const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context ) {
}

//--------------------------------------------------------
void HLTTauDQMSource::endRun( const edm::Run& r, const edm::EventSetup& context ) {
    //Summaries
    for ( unsigned int i = 0; i < summaryPlotters.size(); ++i ) {
        if (summaryPlotters[i]->isValid()) summaryPlotters[i]->plot();
    }
}

//--------------------------------------------------------
void HLTTauDQMSource::endJob() {
}

LVColl HLTTauDQMSource::getFilterCollection( size_t index, int id, const trigger::TriggerEvent& trigEv, double ptCut ) {    
    using namespace trigger;
    //Create output collection
    LVColl out;
    //Fetch all the final trigger objects
    const TriggerObjectCollection& TOC(trigEv.getObjects());
    
    //Filter index
    if ( index != trigEv.sizeFilters() ) {
        const Keys& KEYS = trigEv.filterKeys(index);
        for ( size_t i = 0; i < KEYS.size(); ++i ) {
            const TriggerObject& TO(TOC[KEYS[i]]);
            if( abs(TO.id()) == id ) {
                LV a(TO.px(),TO.py(),TO.pz(),sqrt(TO.px()*TO.px()+TO.py()*TO.py()+TO.pz()*TO.pz()));
                if( a.pt()>ptCut ) out.push_back(a);
            }
	    }
	}
    return out;
}

void HLTTauDQMSource::processPSet( const edm::ParameterSet& pset ) {
    //Get general monitoring parameters
    config_          = pset.getParameter<std::vector<edm::ParameterSet> >("MonitorSetup");
    matching_        = pset.getParameter<edm::ParameterSet>("Matching");
    NPtBins_         = pset.getUntrackedParameter<int>("PtHistoBins",20);
    NEtaBins_        = pset.getUntrackedParameter<int>("EtaHistoBins",20);
    NPhiBins_        = pset.getUntrackedParameter<int>("PhiHistoBins",32);
    EtMax_           = pset.getUntrackedParameter<double>("EtHistoMax",100);
    prescaleEvt_     = pset.getUntrackedParameter<int>("prescaleEvt", -1);
    
    //Clear the plotter collections first
    while (!l1Plotters.empty()) delete l1Plotters.back(), l1Plotters.pop_back();
    while (!caloPlotters.empty()) delete caloPlotters.back(), caloPlotters.pop_back();
    while (!trackPlotters.empty()) delete trackPlotters.back(), trackPlotters.pop_back();
    while (!pathPlotters.empty()) delete pathPlotters.back(), pathPlotters.pop_back();
    while (!litePathPlotters.empty()) delete litePathPlotters.back(), litePathPlotters.pop_back();
    
    //Automatic Configuration
    automation_.AutoCompleteConfig( config_, HLTCP_ );
    automation_.AutoCompleteMatching( matching_, HLTCP_, ".*Ele.*" );
    
    doRefAnalysis_   = matching_.getUntrackedParameter<bool>("doMatching");
    refTriggerEvent_ = matching_.getUntrackedParameter<edm::InputTag>("TriggerEventObject");
    refFilters_      = matching_.getUntrackedParameter<std::vector<edm::ParameterSet> >("matchFilters");
    
    //Read the configuration
    for ( unsigned int i = 0; i < config_.size(); ++i ) {
        std::string configtype;
        try {
            configtype = config_[i].getUntrackedParameter<std::string>("ConfigType");
        } catch ( cms::Exception &e ) {
            edm::LogWarning("HLTTauDQMSource") << e.what() << std::endl;
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
        } else if (configtype == "LitePath") {
            try {
                litePathPlotters.push_back(new HLTTauDQMLitePathPlotter(config_[i],NPtBins_,NEtaBins_,NPhiBins_,EtMax_,doRefAnalysis_,HLTMatchDr_,dqmBaseFolder_));   
            } catch ( cms::Exception &e ) {
                edm::LogWarning("HLTTauDQMSource") << e.what() << std::endl;
                continue;
            }
        }
        summaryPlotters.push_back(new HLTTauDQMSummaryPlotter(config_[i],dqmBaseFolder_));
    }
}
