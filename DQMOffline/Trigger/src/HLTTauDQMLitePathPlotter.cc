#include "DQMOffline/Trigger/interface/HLTTauDQMLitePathPlotter.h"

HLTTauDQMLitePathPlotter::HLTTauDQMLitePathPlotter( const edm::ParameterSet& ps, int etbins, int etabins, int phibins, double maxpt, bool ref, double dr, std::string dqmBaseFolder ) {
    //Initialize Plotter
    name_ = "HLTTauDQMLitePathPlotter";
    
    //Process PSet
    try {
        triggerEvent_    = ps.getUntrackedParameter<edm::InputTag>("TriggerEventObject");
        triggerTag_      = ps.getUntrackedParameter<std::string>("DQMFolder");
        triggerTagAlias_ = ps.getUntrackedParameter<std::string>("Alias","");
        filters_         = ps.getUntrackedParameter<std::vector<edm::ParameterSet> >("Filters");        
        refTauPt_        = ps.getUntrackedParameter<double>("refTauPt",20);
        refLeptonPt_     = ps.getUntrackedParameter<double>("refLeptonPt",15);
        doRefAnalysis_   = ref;
        dqmBaseFolder_   = dqmBaseFolder;
        matchDeltaR_     = dr;
        maxEt_           = maxpt;
        binsEt_          = etbins;
        binsEta_         = etabins;
        binsPhi_         = phibins;
        validity_        = true;
    } catch ( cms::Exception &e ) {
        edm::LogInfo("HLTTauDQMLitePathPlotter::HLTTauDQMLitePathPlotter") << e.what() << std::endl;
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
        
        accepted_events = store_->book1D("PathTriggerBits","Accepted Events per Path;;entries",filterObjs_.size(),0,filterObjs_.size());
        
        for ( size_t k = 0; k < filterObjs_.size(); ++k ) {
            accepted_events->setBinLabel(k+1,filterObjs_[k].getAlias(),1);
            if ( filterObjs_[k].getNTriggeredTaus() >= 2 || (filterObjs_[k].getNTriggeredTaus() >= 1 && filterObjs_[k].getNTriggeredLeptons() >= 1) ) { 
                mass_distribution.insert(std::make_pair(filterObjs_[k].getAlias(), store_->book1D(("mass_"+filterObjs_[k].getAlias()).c_str(),("Mass Distribution for "+filterObjs_[k].getAlias()).c_str(),100,0,500))); 
            }
        }
        
        if (doRefAnalysis_) {
            accepted_events_matched = store_->book1D("MatchedPathTriggerBits","Accepted+Matched Events per Path;;entries",filterObjs_.size(),0,filterObjs_.size());
            accepted_events_matched->getTH1F()->Sumw2();
            
            for ( size_t k = 0; k < filterObjs_.size(); ++k ) {
                accepted_events_matched->setBinLabel(k+1,filterObjs_[k].getAlias(),1);
            }
            ref_events = store_->book1D("RefEvents","Reference Events per Path",filterObjs_.size(),0,filterObjs_.size());
            ref_events->getTH1F()->Sumw2();
            
            for ( size_t k = 0; k < filterObjs_.size(); ++k ) {
                ref_events->setBinLabel(k+1,filterObjs_[k].getAlias(),1);
            }
        }
        
        tauEt = store_->book1D("TrigTauEt","#tau E_{t}",binsEt_,0,maxEt_);
        tauEta = store_->book1D("TrigTauEta","#tau #eta",binsEta_,-2.5,2.5);
        tauPhi = store_->book1D("TrigTauPhi","#tau #phi",binsPhi_,-3.2,3.2);
        
        if (doRefAnalysis_) {
            store_->setCurrentFolder(triggerTag()+"/EfficiencyHelpers");
            store_->removeContents();
            
            tauEtEffNum = store_->book1D("TrigTauEtEffNum","#tau E_{T} Efficiency",binsEt_,0,maxEt_);
            tauEtEffNum->getTH1F()->Sumw2();
            
            tauEtEffDenom = store_->book1D("TrigTauEtEffDenom","#tau E_{T} Denominator",binsEt_,0,maxEt_);
            tauEtEffDenom->getTH1F()->Sumw2();
            
            tauEtaEffNum = store_->book1D("TrigTauEtaEffNum","#tau #eta Efficiency",binsEta_,-2.5,2.5);
            tauEtaEffNum->getTH1F()->Sumw2();
            
            tauEtaEffDenom = store_->book1D("TrigTauEtaEffDenom","#tau #eta Denominator",binsEta_,-2.5,2.5);
            tauEtaEffDenom->getTH1F()->Sumw2();
            
            tauPhiEffNum = store_->book1D("TrigTauPhiEffNum","#tau #phi Efficiency",binsPhi_,-3.2,3.2);
            tauPhiEffNum->getTH1F()->Sumw2();
            
            tauPhiEffDenom = store_->book1D("TrigTauPhiEffDenom","#tau #phi Denominator",binsPhi_,-3.2,3.2);
            tauPhiEffDenom->getTH1F()->Sumw2();
        }
    }
}

HLTTauDQMLitePathPlotter::~HLTTauDQMLitePathPlotter() {
}

//
// member functions
//

void HLTTauDQMLitePathPlotter::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::map<int,LVColl>& refC ) {
    std::vector<bool> isGoodReferenceEvent;
    
    LV highestRefTau(0.,0.,0.,0.0001);
    LVColl triggeredTaus, refTaus, refElectrons, refMuons;
    
    std::map<int,LVColl>::const_iterator iref;
    
    //Tau reference
    iref = refC.find(15);
    if ( iref != refC.end() ) {
        for ( LVColl::const_iterator lvi = iref->second.begin(); lvi != iref->second.end(); ++lvi ) {
            if ( lvi->Et() > highestRefTau.pt() ) {
                highestRefTau = *lvi;
            }
        }
    }
	
    //Fill ref collection for the filters
    if (doRefAnalysis_) {
        unsigned int highPtTaus = 0;
        unsigned int highPtElectrons = 0;
        unsigned int highPtMuons = 0;
        
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
        //Electron reference
        iref = refC.find(11);
        if ( iref != refC.end() ) {
            for ( LVColl::const_iterator lvi = iref->second.begin(); lvi != iref->second.end(); ++lvi ) {
                if ( lvi->Et() > refLeptonPt_ ) {
                    highPtElectrons++;
                }
                refElectrons.push_back(*lvi);
            }
        }
        //Muon reference
        iref = refC.find(13);
        if ( iref != refC.end() ) {
            for ( LVColl::const_iterator lvi = iref->second.begin(); lvi != iref->second.end(); ++lvi ) {
                if ( lvi->Et() > refLeptonPt_ ) {
                    highPtMuons++;
                }
                refMuons.push_back(*lvi);
            }
        }
        
        for ( size_t i = 0; i < filterObjs_.size(); ++i ) {
            bool tau_ok = true;
            bool leptons_ok = true;
            
            if ( highPtTaus < filterObjs_[i].getNTriggeredTaus() ) tau_ok = false;
            if ( filterObjs_[i].leptonId() == 11 && highPtElectrons < filterObjs_[i].getNTriggeredLeptons() ) leptons_ok = false;
            if ( filterObjs_[i].leptonId() == 13 && highPtMuons < filterObjs_[i].getNTriggeredLeptons() ) leptons_ok = false;

            if ( tau_ok && leptons_ok ) {
                ref_events->Fill(i+0.5);
                isGoodReferenceEvent.push_back(true);
            } else {
                isGoodReferenceEvent.push_back(false);
            }
        }
    }
    
    //Get the TriggerEvent
    edm::Handle<trigger::TriggerEvent> trigEv;
    bool gotTEV = iEvent.getByLabel(triggerEvent_,trigEv) && trigEv.isValid();
    
    if ( gotTEV ) {
        //Loop through the filters
        for ( size_t i = 0; i < filterObjs_.size(); ++i ) {
            std::map<std::string, MonitorElement*>::iterator thisMassDist = mass_distribution.find(filterObjs_[i].getAlias());
            size_t ID = trigEv->filterIndex(filterObjs_[i].getFilterName());
            if ( ID != trigEv->sizeFilters() ) {
                LVColl leptons = getFilterCollection(ID,filterObjs_[i].getLeptonType(),*trigEv);
                LVColl taus = getFilterCollection(ID,filterObjs_[i].getTauType(),*trigEv);
                
                //If this is the single tau trigger copy the collection for the turn on
                if ( filterObjs_[i].getNTriggeredTaus() == 1 && filterObjs_[i].getNTriggeredLeptons() == 0 ) {
                    triggeredTaus = taus;
                }
                
                //Fired
                if ( leptons.size() >= filterObjs_[i].getNTriggeredLeptons() && taus.size() >= filterObjs_[i].getNTriggeredTaus() ) {
                    accepted_events->Fill(i+0.5);
                    
                    //Now do the matching only though if we have a good reference event
                    if ( doRefAnalysis_ ) {
                        if ( isGoodReferenceEvent.at(i) ) {
                            size_t nT = 0;
                            for ( size_t j = 0; j < taus.size(); ++j ) {
                                if (match(taus[j],refTaus,matchDeltaR_).first) nT++;
                            }
                            size_t nL = 0;
                            for ( size_t j = 0; j < leptons.size(); ++j ) {
                                if (match(leptons[j],refElectrons,matchDeltaR_).first) nL++;
                                if (match(leptons[j],refMuons,matchDeltaR_).first) nL++;
                            }
                            if ( nT >= filterObjs_[i].getNTriggeredTaus() && nL >= filterObjs_[i].getNTriggeredLeptons() ) {
                                accepted_events_matched->Fill(i+0.5);
                                if ( filterObjs_[i].getNTriggeredTaus() >= 2 && refTaus.size() >= 2 ) {
                                    if (thisMassDist != mass_distribution.end()) thisMassDist->second->Fill( (refTaus[0]+refTaus[1]).M() );
                                } else if ( filterObjs_[i].getNTriggeredTaus() >= 1 && filterObjs_[i].getNTriggeredLeptons() >= 1 ) {
                                    if ( filterObjs_[i].leptonId() == 11 && refElectrons.size() >= 1 ) {
                                        if (thisMassDist != mass_distribution.end()) thisMassDist->second->Fill( (refTaus[0]+refElectrons[0]).M() );
                                    }				      
                                    if ( filterObjs_[i].leptonId() == 13 && refMuons.size() >= 1 ) {
                                        if (thisMassDist != mass_distribution.end()) thisMassDist->second->Fill( (refTaus[0]+refMuons[0]).M() );
                                    }
                                }
                            }
                        }
                    } else {
                        if ( filterObjs_[i].getNTriggeredTaus() >= 2 ) {
                            if (thisMassDist != mass_distribution.end()) thisMassDist->second->Fill( (taus[0]+taus[1]).M() );
                        } else if ( filterObjs_[i].getNTriggeredTaus() >= 1 && filterObjs_[i].getNTriggeredLeptons() >= 1 ) {
                            if (thisMassDist != mass_distribution.end()) thisMassDist->second->Fill( (taus[0]+leptons[0]).M() );
                        }
                    }
                }
            }
        }
        
        //Do the object thing for the highest tau!
        if ( triggeredTaus.size() > 0 ) {
            LV highestTriggeredTau(0.,0.,0.,0.0001);
            for ( unsigned int i = 0; i < triggeredTaus.size(); ++i ) {
                if ( triggeredTaus[i].pt() > highestTriggeredTau.pt() ) {
                    highestTriggeredTau = triggeredTaus[i];
                }
            }
            
            tauEt->Fill(highestTriggeredTau.pt());
            tauEta->Fill(highestTriggeredTau.eta());
            tauPhi->Fill(highestTriggeredTau.phi());
            
            if ( doRefAnalysis_ && highestRefTau.pt() > 5.0 ) {
                tauEtEffDenom->Fill(highestRefTau.pt());
                tauEtaEffDenom->Fill(highestRefTau.eta());
                tauPhiEffDenom->Fill(highestRefTau.phi());
                
                if ( ROOT::Math::VectorUtil::DeltaR(highestRefTau,highestTriggeredTau) < matchDeltaR_ ) {
                    tauEtEffNum->Fill(highestRefTau.pt());
                    tauEtaEffNum->Fill(highestRefTau.eta());
                    tauPhiEffNum->Fill(highestRefTau.phi());
                }
            }
        }
    }
}

LVColl HLTTauDQMLitePathPlotter::getObjectCollection( int id, const trigger::TriggerEvent& trigEv ) {
    trigger::TriggerObjectCollection triggerObjects;
	triggerObjects = trigEv.getObjects();
    
	LVColl out;
	for ( unsigned int i = 0; i < triggerObjects.size(); ++i ) {
        if ( abs(triggerObjects[i].id()) == id ) {
            LV a(triggerObjects[i].px(),triggerObjects[i].py(),triggerObjects[i].pz(),triggerObjects[i].energy());
            out.push_back(a);
        }
    }
	return out;
}

LVColl HLTTauDQMLitePathPlotter::getFilterCollection( size_t index, int id, const trigger::TriggerEvent& trigEv ) {
    //Create output collection
    LVColl out;
    //Get all the final trigger objects
    const trigger::TriggerObjectCollection& TOC(trigEv.getObjects());
    
    //Filter index
    if ( index != trigEv.sizeFilters() ) {
        const trigger::Keys& KEYS = trigEv.filterKeys(index);
        for ( size_t i = 0; i < KEYS.size(); ++i ) {
            const trigger::TriggerObject& TO(TOC[KEYS[i]]);
            LV a(TO.px(),TO.py(),TO.pz(),sqrt(TO.px()*TO.px()+TO.py()*TO.py()+TO.pz()*TO.pz()));
            out.push_back(a);
	    }
	}
    return out;
}
