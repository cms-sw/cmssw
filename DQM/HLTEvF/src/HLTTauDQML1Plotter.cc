#include "DQM/HLTEvF/interface/HLTTauDQML1Plotter.h"

HLTTauDQML1Plotter::HLTTauDQML1Plotter( const edm::ParameterSet& ps, int etbins, int etabins, int phibins, double maxpt, bool ref, double dr, std::string dqmBaseFolder ) {
    //Initialize Plotter
    name_ = "HLTTauDQML1Plotter";
    
    //Process PSet
    try {
        triggerTag_       = ps.getUntrackedParameter<std::string>("DQMFolder");
        triggerTagAlias_  = ps.getUntrackedParameter<std::string>("Alias","");
        l1ExtraTaus_      = ps.getParameter<edm::InputTag>("L1Taus");
        l1ExtraJets_      = ps.getParameter<edm::InputTag>("L1Jets");
        l1ExtraElectrons_ = ps.getParameter<edm::InputTag>("L1Electrons");
        l1ExtraMuons_     = ps.getParameter<edm::InputTag>("L1Muons");
        doRefAnalysis_    = ref;
        dqmBaseFolder_    = dqmBaseFolder;
        matchDeltaR_      = dr;
        maxEt_            = maxpt;
        binsEt_           = etbins;
        binsEta_          = etabins;
        binsPhi_          = phibins;
        validity_         = true;
    } catch ( cms::Exception &e ) {
        edm::LogWarning("HLTTauDQML1Plotter::HLTTauDQML1Plotter") << e.what() << std::endl;
        validity_ = false;
        return;
    }
    
    if (store_) {
        //Create the histograms
        store_->setCurrentFolder(triggerTag());
        store_->removeContents();
        
        l1tauEt_ = store_->book1D("L1TauEt","L1 #tau E_{T};L1 #tau E_{T};entries",binsEt_,0,maxEt_);
        l1tauEta_ = store_->book1D("L1TauEta","L1 #tau #eta;L1 #tau #eta;entries",binsEta_,-2.5,2.5);
        l1tauPhi_ = store_->book1D("L1TauPhi","L1 #tau #phi;L1 #tau #phi;entries",binsPhi_,-3.2,3.2);
        
        l1jetEt_ = store_->book1D("L1JetEt","L1 jet E_{T};L1 Central Jet E_{T};entries",binsEt_,0,maxEt_);
        l1jetEta_ = store_->book1D("L1JetEta","L1 jet #eta;L1 Central Jet #eta;entries",binsEta_,-2.5,2.5);
        l1jetPhi_ = store_->book1D("L1JetPhi","L1 jet #phi;L1 Central Jet #phi;entries",binsPhi_,-3.2,3.2);
        
        inputEvents_ = store_->book1D("InputEvents","Events Read;;entries",2,0,2);
        
        l1electronEt_ = store_->book1D("L1ElectronEt","L1 electron E_{T};L1 e/#gamma  E_{T};entries",binsEt_,0,maxEt_);
        l1electronEta_ = store_->book1D("L1ElectronEta","L1 electron #eta;L1 e/#gamma  #eta;entries",binsEta_,-2.5,2.5);
        l1electronPhi_ = store_->book1D("L1ElectronPhi","L1 electron #phi;L1 e/#gamma  #phi;entries",binsPhi_,-3.2,3.2);
        
        l1muonEt_ = store_->book1D("L1MuonEt","L1 muon p_{T};L1 #mu p_{T};entries",binsEt_,0,maxEt_);
        l1muonEta_ = store_->book1D("L1MuonEta","L1 muon #eta;L1 #mu #eta;entries",binsEta_,-2.5,2.5);
        l1muonPhi_ = store_->book1D("L1MuonPhi","L1 muon #phi;L1 #mu #phi;entries",binsPhi_,-3.2,3.2);
        
        l1doubleTauPath_ = store_->book2D("L1DoubleTau","L1 Double Tau Path E_{T};first L1 #tau p_{T};second L1 #tau p_{T}",binsEt_,0,maxEt_,binsEt_,0,maxEt_);
        l1muonTauPath_ = store_->book2D("L1MuonTau","L1 Muon Tau Path E_{T};first L1 #tau p_{T};first L1 #gamma p_{T}",binsEt_,0,maxEt_,binsEt_,0,maxEt_);
        l1electronTauPath_ = store_->book2D("L1ElectronTau","L1 Electron Tau Path E_{T};first L1 #mu p_{T};second L1 #mu p_{T}",binsEt_,0,maxEt_,binsEt_,0,maxEt_);
        
        firstTauEt_ = store_->book1D("L1LeadTauEt","L1 lead #tau E_{T}",binsEt_,0,maxEt_);
        firstTauEt_->getTH1F()->Sumw2();
        
        secondTauEt_ = store_->book1D("L1SecondTauEt","L1 second #tau E_{T}",binsEt_,0,maxEt_);
        secondTauEt_->getTH1F()->Sumw2();
        
        if (doRefAnalysis_) {
            l1tauEtRes_ = store_->book1D("L1TauEtResol","L1 #tau E_{T} resolution;[L1 #tau E_{T}-Ref #tau E_{T}]/Ref #tau E_{T};entries",40,-2,2);
            
            store_->setCurrentFolder(triggerTag()+"/EfficiencyHelpers");
            store_->removeContents();
            
            l1tauEtEffNum_ = store_->book1D("L1TauEtEffNum","L1 #tau E_{T} Efficiency Numerator",binsEt_,0,maxEt_);
            l1tauEtEffNum_->getTH1F()->Sumw2();
            
            l1tauEtEffDenom_ = store_->book1D("L1TauEtEffDenom","L1 #tau E_{T} Denominator",binsEt_,0,maxEt_);
            l1tauEtEffDenom_->getTH1F()->Sumw2();
            
            l1tauEtaEffNum_ = store_->book1D("L1TauEtaEffNum","L1 #tau #eta Efficiency",binsEta_,-2.5,2.5);
            l1tauEtaEffNum_->getTH1F()->Sumw2();
            
            l1tauEtaEffDenom_ = store_->book1D("L1TauEtaEffDenom","L1 #tau #eta Denominator",binsEta_,-2.5,2.5);
            l1tauEtaEffDenom_->getTH1F()->Sumw2();
            
            l1tauPhiEffNum_ = store_->book1D("L1TauPhiEffNum","L1 #tau #phi Efficiency",binsPhi_,-3.2,3.2);
            l1tauPhiEffNum_->getTH1F()->Sumw2();
            
            l1tauPhiEffDenom_ = store_->book1D("L1TauPhiEffDenom","L1 #tau #phi Denominator",binsPhi_,-3.2,3.2);
            l1tauPhiEffDenom_->getTH1F()->Sumw2();
            
            l1jetEtEffNum_ = store_->book1D("L1JetEtEffNum","L1 jet E_{T} Efficiency",binsEt_,0,maxEt_);
            l1jetEtEffNum_->getTH1F()->Sumw2();
            
            l1jetEtEffDenom_ = store_->book1D("L1JetEtEffDenom","L1 jet E_{T} Denominator",binsEt_,0,maxEt_);
            l1jetEtEffDenom_->getTH1F()->Sumw2();
            
            l1jetEtaEffNum_ = store_->book1D("L1JetEtaEffNum","L1 jet #eta Efficiency",binsEta_,-2.5,2.5);
            l1jetEtaEffNum_->getTH1F()->Sumw2();
            
            l1jetEtaEffDenom_ = store_->book1D("L1JetEtaEffDenom","L1 jet #eta Denominator",binsEta_,-2.5,2.5);
            l1jetEtaEffDenom_->getTH1F()->Sumw2();
            
            l1jetPhiEffNum_ = store_->book1D("L1JetPhiEffNum","L1 jet #phi Efficiency",binsPhi_,-3.2,3.2);
            l1jetPhiEffNum_->getTH1F()->Sumw2();
            
            l1jetPhiEffDenom_ = store_->book1D("L1JetPhiEffDenom","L1 jet #phi Denominator",binsPhi_,-3.2,3.2);
            l1jetPhiEffDenom_->getTH1F()->Sumw2();
        }
    }
}

HLTTauDQML1Plotter::~HLTTauDQML1Plotter() {
}

//
// member functions
//

void HLTTauDQML1Plotter::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::map<int,LVColl>& refC ) {
    LVColl refTaus, refElectrons, refMuons;
    
    if ( doRefAnalysis_ ) {
        std::map<int,LVColl>::const_iterator iref;

        //Tau reference
        iref = refC.find(15);
        if ( iref != refC.end() ) refTaus = iref->second;
        
        //Electron reference
        iref = refC.find(11);
        if ( iref != refC.end() ) refElectrons = iref->second;
        
        //Muon reference
        iref = refC.find(13);
        if ( iref != refC.end() ) refMuons = iref->second;
        
        for ( LVColl::const_iterator iter = refTaus.begin(); iter != refTaus.end(); ++iter ) {
            l1tauEtEffDenom_->Fill(iter->pt());
            l1jetEtEffDenom_->Fill(iter->pt());
            
            l1tauEtaEffDenom_->Fill(iter->eta());
            l1jetEtaEffDenom_->Fill(iter->eta());
            
            l1tauPhiEffDenom_->Fill(iter->phi());
            l1jetPhiEffDenom_->Fill(iter->phi());
        }
    }
    
    //Analyze L1 Objects (Tau+Jets)
    edm::Handle<l1extra::L1JetParticleCollection> taus;
    edm::Handle<l1extra::L1JetParticleCollection> jets;
    edm::Handle<l1extra::L1EmParticleCollection> electrons;
    edm::Handle<l1extra::L1MuonParticleCollection> muons;
    
    LVColl pathTaus;
    LVColl pathMuons;
    LVColl pathElectrons;
    
    //Set Variables for the threshold plot
    LVColl l1taus;
    LVColl l1electrons;
    LVColl l1muons;
    LVColl l1jets;

    bool gotL1Taus = iEvent.getByLabel(l1ExtraTaus_,taus) && taus.isValid();
    bool gotL1Jets = iEvent.getByLabel(l1ExtraJets_,jets) && jets.isValid();
    bool gotL1Electrons = iEvent.getByLabel(l1ExtraElectrons_,electrons) && electrons.isValid();
    bool gotL1Muons = iEvent.getByLabel(l1ExtraMuons_,muons) && muons.isValid();
    
    if ( gotL1Taus ) {
        if ( taus->size() > 0 ) {
            if ( !doRefAnalysis_ ) {
                firstTauEt_->Fill((*taus)[0].pt());
                if ( taus->size() > 1 ) secondTauEt_->Fill((*taus)[0].pt());
            }
            for ( l1extra::L1JetParticleCollection::const_iterator i = taus->begin(); i != taus->end(); ++i ) {
                l1taus.push_back(i->p4());
                if ( !doRefAnalysis_ ) {
                    l1tauEt_->Fill(i->et());
                    l1tauEta_->Fill(i->eta());
                    l1tauPhi_->Fill(i->phi());
                    pathTaus.push_back(i->p4());
                }
            }
        }
    }
    if ( gotL1Jets ) {
        if ( jets->size() > 0 ) {
            for( l1extra::L1JetParticleCollection::const_iterator i = jets->begin(); i != jets->end(); ++i ) {	
                l1jets.push_back(i->p4());
                if ( !doRefAnalysis_ ) {
                    l1jetEt_->Fill(i->et());
                    l1jetEta_->Fill(i->eta());
                    l1jetPhi_->Fill(i->phi());
                    pathTaus.push_back(i->p4());
                }
            }
        }
    }
    if ( gotL1Electrons ) {
        if( electrons->size() > 0 ) {
            for ( l1extra::L1EmParticleCollection::const_iterator i = electrons->begin(); i != electrons->end(); ++i ) {
                l1electrons.push_back(i->p4());
                l1electronEt_->Fill(i->et());
                l1electronEta_->Fill(i->eta());
                l1electronPhi_->Fill(i->phi());
                pathElectrons.push_back(i->p4());
            }
        }
    }
    if ( gotL1Muons ) {
        if ( muons->size() > 0 ) {
            for ( l1extra::L1MuonParticleCollection::const_iterator i = muons->begin(); i != muons->end(); ++i ) {
                l1muons.push_back(i->p4());
                l1muonEt_->Fill(i->et());
                l1muonEta_->Fill(i->eta());
                l1muonPhi_->Fill(i->phi());
                pathMuons.push_back(i->p4());
            }
        }
    }
    
    //Now do the efficiency matching
    if ( doRefAnalysis_ ) {
        for ( LVColl::const_iterator i = refTaus.begin(); i != refTaus.end(); ++i ) {
            std::pair<bool,LV> m = match(*i,l1taus,matchDeltaR_);
            if ( m.first ) {
                l1tauEt_->Fill(m.second.pt());
                l1tauEta_->Fill(m.second.eta());
                l1tauPhi_->Fill(m.second.phi());
                l1tauEtEffNum_->Fill(i->pt());
                l1tauEtaEffNum_->Fill(i->eta());
                l1tauPhiEffNum_->Fill(i->phi());
                l1tauEtRes_->Fill((m.second.pt()-i->pt())/i->pt());
                pathTaus.push_back(m.second);
            }
        }
        
        for ( LVColl::const_iterator i = refTaus.begin(); i != refTaus.end(); ++i ) {
            std::pair<bool,LV> m = match(*i,l1jets,matchDeltaR_);
            if ( m.first ) {
                l1jetEt_->Fill(m.second.pt());
                l1jetEta_->Fill(m.second.eta());
                l1jetPhi_->Fill(m.second.phi());
                l1jetEtEffNum_->Fill(i->pt());
                l1jetEtaEffNum_->Fill(i->eta());
                l1jetPhiEffNum_->Fill(i->phi());
            }
        }
        
        for ( LVColl::const_iterator i = refElectrons.begin(); i != refElectrons.end(); ++i ) {
            std::pair<bool,LV> m = match(*i,l1electrons,matchDeltaR_);
            if( m.first ) {
                l1electronEt_->Fill(m.second.pt());
                l1electronEta_->Fill(m.second.eta());
                l1electronPhi_->Fill(m.second.phi());
                pathElectrons.push_back(m.second);
            }
        }
        
        for ( LVColl::const_iterator i = refMuons.begin(); i != refMuons.end(); ++i ) {
            std::pair<bool,LV> m = match(*i,l1muons,matchDeltaR_);
            if ( m.first ) {
                l1muonEt_->Fill(m.second.pt());
                l1muonEta_->Fill(m.second.eta());
                l1muonPhi_->Fill(m.second.phi());
                pathMuons.push_back(m.second);
            }
        }
    }
    
    
    //Fill the Threshold Monitoring
    if(pathTaus.size() > 1) std::sort(pathTaus.begin(),pathTaus.end(),ptSort);
    if(pathElectrons.size() > 1) std::sort(pathElectrons.begin(),pathElectrons.end(),ptSort);
    if(pathMuons.size() > 1) std::sort(pathMuons.begin(),pathMuons.end(),ptSort);
    
    if ( pathTaus.size() > 0 ) {
        firstTauEt_->Fill(pathTaus[0].pt());
        inputEvents_->Fill(0.5);
    }
    if ( pathTaus.size() > 1 ) {
        secondTauEt_->Fill(pathTaus[1].pt());
        inputEvents_->Fill(1.5);
    }
    if ( pathTaus.size() >= 2 ) {
        l1doubleTauPath_->Fill(pathTaus[0].pt(),pathTaus[1].pt());
    }
    if ( pathTaus.size() >= 1 && pathElectrons.size() >= 1 ) {
        l1electronTauPath_->Fill(pathTaus[0].pt(),pathElectrons[0].pt());
    }
    if ( pathTaus.size() >= 1 && pathMuons.size() >= 1 ) {
        l1muonTauPath_->Fill(pathTaus[0].pt(),pathMuons[0].pt());
    }
}
