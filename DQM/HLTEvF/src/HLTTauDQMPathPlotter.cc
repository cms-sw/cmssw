#include "DQM/HLTEvF/interface/HLTTauDQMPathPlotter.h"

HLTTauDQMPathPlotter::HLTTauDQMPathPlotter( const edm::ParameterSet& ps, bool ref, std::string dqmBaseFolder ) {
    //Initialize Plotter
    name_ = "HLTTauDQMPathPlotter";
    
    //Process PSet
    try {
        triggerEventObject_   = ps.getUntrackedParameter<edm::InputTag>("TriggerEventObject");
        triggerTag_           = ps.getUntrackedParameter<std::string>("DQMFolder");
        triggerTagAlias_      = ps.getUntrackedParameter<std::string>("Alias","");
        filters_              = ps.getUntrackedParameter<std::vector<edm::ParameterSet> >("Filters");
        reference_            = ps.getUntrackedParameter<edm::ParameterSet>("Reference");
        refNTriggeredTaus_    = reference_.getUntrackedParameter<unsigned int>("NTriggeredTaus");
        refNTriggeredLeptons_ = reference_.getUntrackedParameter<unsigned int>("NTriggeredLeptons");
        refTauPt_             = reference_.getUntrackedParameter<double>("refTauPt",20);
        refLeptonPt_          = reference_.getUntrackedParameter<double>("refLeptonPt",15);
        dqmBaseFolder_        = dqmBaseFolder;
        doRefAnalysis_        = ref;
        validity_             = true;
    } catch ( cms::Exception &e ) {
        edm::LogInfo("HLTTauDQMPathPlotter::HLTTauDQMPathPlotter") << e.what() << std::endl;
        validity_ = false;
        return;
    }
    
    for ( std::vector<edm::ParameterSet>::const_iterator iter = filters_.begin(); iter != filters_.end(); ++iter ) {
        HLTTauDQMPlotter::FilterObject tmp(*iter);
        if (tmp.isValid()) filterObjs_.push_back(tmp);
    }
    
    if (store_) {
        //Create the histograms
        store_->setCurrentFolder(triggerTag());
        store_->removeContents();
        
        accepted_events = store_->book1D("TriggerBits","Accepted Events per Path;;entries",filterObjs_.size(),0,filterObjs_.size());
        for ( size_t k = 0; k < filterObjs_.size(); ++k ) {
            accepted_events->setBinLabel(k+1,filterObjs_[k].getAlias(),1);
        }
        if (doRefAnalysis_) {
            accepted_events_matched = store_->book1D("MatchedTriggerBits","Accepted+Matched Events per Path;;entries",filterObjs_.size()+1,0,filterObjs_.size()+1);
            accepted_events_matched->setBinLabel(1,"RefEvents",1);
            for ( size_t k = 0; k < filterObjs_.size(); ++k ) {
                accepted_events_matched->setBinLabel(k+2,filterObjs_[k].getAlias(),1);
            }
        }
    }
}

HLTTauDQMPathPlotter::~HLTTauDQMPathPlotter() {
}

//
// member functions
//

void HLTTauDQMPathPlotter::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::map<int,LVColl>& refC ) {
    using namespace std;
    using namespace edm;
    using namespace reco;
    using namespace l1extra;
    using namespace trigger;
    
    bool isGoodReferenceEvent = false;
    LVColl refTaus, refLeptons;
    
    if (doRefAnalysis_) {
        unsigned int highPtTaus = 0;
        unsigned int highPtElectrons = 0;
        unsigned int highPtMuons = 0;
        
        bool tau_ok = true;
        bool leptons_ok = true;
        
        std::map<int,LVColl>::const_iterator iref;
        
        //Tau reference
        iref = refC.find(15);
        if ( iref != refC.end() ) {
            for ( LVColl::const_iterator lvi = iref->second.begin(); lvi != iref->second.end(); ++lvi ) {
                if ( lvi->Et() > refTauPt_ ) {
                    highPtTaus++;
                }
                refTaus.push_back(*lvi);
            }
        }
        if ( highPtTaus < refNTriggeredTaus_ ) tau_ok = false;
        
        //Electron reference
        iref = refC.find(11);
        if ( iref != refC.end() ) {
            for ( LVColl::const_iterator lvi = iref->second.begin(); lvi != iref->second.end(); ++lvi ) {
                if ( lvi->Et() > refLeptonPt_ ) {
                    highPtElectrons++;
                }
                refLeptons.push_back(*lvi);
            }
        }
        if ( filterObjs_.back().leptonId() == 11 && highPtElectrons < refNTriggeredLeptons_ ) leptons_ok = false;
        
        //Muon reference
        iref = refC.find(13);
        if ( iref != refC.end() ) {
            for ( LVColl::const_iterator lvi = iref->second.begin(); lvi != iref->second.end(); ++lvi ) {
                if ( lvi->Et() > refLeptonPt_ ) {
                    highPtMuons++;
                }
                refLeptons.push_back(*lvi);
            }
        }
        if ( filterObjs_.back().leptonId() == 13 && highPtMuons < refNTriggeredLeptons_ ) leptons_ok = false;
        
        if ( tau_ok && leptons_ok ) {
            accepted_events_matched->Fill(0.5);
            isGoodReferenceEvent = true;
        }
    }
    
    Handle<TriggerEventWithRefs> trigEv;
    bool gotTEV = iEvent.getByLabel(triggerEventObject_,trigEv) && trigEv.isValid();
    
    if (gotTEV) {
        //Loop through the filters
        for ( size_t i = 0; i < filterObjs_.size(); ++i ) {		
            size_t ID = trigEv->filterIndex(filterObjs_[i].getFilterName());
            
            if ( ID != trigEv->size() ) {
                LVColl leptons = getFilterCollection(ID,filterObjs_[i].getLeptonType(),*trigEv);
                LVColl taus = getFilterCollection(ID,filterObjs_[i].getTauType(),*trigEv);
                //Fired
                if ( leptons.size() >= filterObjs_[i].getNTriggeredLeptons() && taus.size() >= filterObjs_[i].getNTriggeredTaus() ) {
                    accepted_events->Fill(i+0.5);
                    //Now do the matching only though if we have a good reference event
                    if ( doRefAnalysis_ && isGoodReferenceEvent ) {
                        size_t nT = 0;
                        for ( size_t j = 0; j < taus.size(); ++j ) {
                            if( match(taus[j],refTaus,filterObjs_[i].getMatchDeltaR()).first ) nT++;
                        }
                        
                        size_t nL = 0;
                        for ( size_t j = 0; j < leptons.size(); ++j ) {
                            if ( match(leptons[j],refLeptons,filterObjs_[i].getMatchDeltaR()).first ) nL++;
                        }
                        
                        if ( nT >= filterObjs_[i].getNTriggeredTaus() && nL >= filterObjs_[i].getNTriggeredLeptons() ) {
                            accepted_events_matched->Fill(i+1.5);
                        }
                    }
                }
            }
        }
    }
}

LVColl HLTTauDQMPathPlotter::getFilterCollection( size_t filterID, int id, const trigger::TriggerEventWithRefs& trigEv ) {
    using namespace trigger;
    
    LVColl out;
    
    if ( id == trigger::TriggerL1IsoEG || id == trigger::TriggerL1NoIsoEG ) {
        VRl1em obj;
        trigEv.getObjects(filterID,id,obj);
        for (size_t i=0;i<obj.size();++i)
            if (obj.at(i).isAvailable())
                out.push_back(obj[i]->p4());
    }
    
    if ( id == trigger::TriggerL1Mu ) {
        VRl1muon obj;
        trigEv.getObjects(filterID,id,obj);
        for (size_t i=0;i<obj.size();++i)
            if (obj.at(i).isAvailable())
                out.push_back(obj[i]->p4());
    }
    
    if ( id == trigger::TriggerMuon ) {
        VRmuon obj;
        trigEv.getObjects(filterID,id,obj);
        for (size_t i=0;i<obj.size();++i)
            if (obj.at(i).isAvailable())
                out.push_back(obj[i]->p4());
    }
    
    if ( id == trigger::TriggerElectron ) {
        VRelectron obj;
        trigEv.getObjects(filterID,id,obj);
        for (size_t i=0;i<obj.size();++i)
            if (obj.at(i).isAvailable())
                out.push_back(obj[i]->p4());
    }
    
    if ( id == trigger::TriggerL1TauJet ) {
        VRl1jet obj;
        trigEv.getObjects(filterID,id,obj);
        for (size_t i=0;i<obj.size();++i)
            if (obj.at(i).isAvailable())
                out.push_back(obj[i]->p4());
        trigEv.getObjects(filterID,trigger::TriggerL1CenJet,obj);
        for (size_t i=0;i<obj.size();++i)
            if (obj.at(i).isAvailable())
                out.push_back(obj[i]->p4());
        
    }
    
    if ( id == trigger::TriggerTau ) {
        VRjet obj;
        trigEv.getObjects(filterID,id,obj);
        for (size_t i=0;i<obj.size();++i)
            if (obj.at(i).isAvailable())
                out.push_back(obj[i]->p4());
    }
    
    return out;
}
