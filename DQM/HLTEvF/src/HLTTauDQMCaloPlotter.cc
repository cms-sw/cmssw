#include "DQM/HLTEvF/interface/HLTTauDQMCaloPlotter.h"

HLTTauDQMCaloPlotter::HLTTauDQMCaloPlotter( const edm::ParameterSet& iConfig, int etbins, int etabins, int phibins, double maxpt, bool ref, double dr, std::string dqmBaseFolder ) {
    //Initialize Plotter
    name_ = "HLTTauDQMCaloPlotter";
    
    //Process PSet
    try {
        l2preJets_       = iConfig.getUntrackedParameter<std::vector<edm::InputTag> >("L2RegionalJets");
        l2TauInfoAssoc_  = iConfig.getUntrackedParameter<edm::InputTag>("L2InfoAssociationInput");
        triggerTag_      = iConfig.getUntrackedParameter<std::string>("DQMFolder");
        triggerTagAlias_ = iConfig.getUntrackedParameter<std::string>("Alias","");
        l2Isolated_      = iConfig.getUntrackedParameter<edm::InputTag>("L2IsolatedJets");
        matchDeltaRMC_   = dr;
        EtMax_           = maxpt;
        NPtBins_         = etbins;
        NEtaBins_        = etabins;
        NPhiBins_        = phibins;
        doRef_           = ref;
        dqmBaseFolder_   = dqmBaseFolder;
        validity_        = true;
    } catch ( cms::Exception &e ) {
        edm::LogInfo("HLTTauDQMCaloPlotter::HLTTauDQMCaloPlotter") << e.what() << std::endl;
        validity_ = false;
        return;
    }
    
    if (store_) {
        //Create the histograms
        store_->setCurrentFolder(triggerTag());
        store_->removeContents();
        
        preJetEt = store_->book1D("L2PreTauEt","L2 regional #tau E_{t};L2 regional Jet E_{T};entries",NPtBins_,0,EtMax_);
        preJetEta = store_->book1D("L2PreTauEta","L2 regional #tau #eta;L2 regional Jet #eta;entries",NEtaBins_,-2.5,2.5);
        preJetPhi = store_->book1D("L2PreTauPhi","L2 regional #tau #phi;L2 regional Jet #phi;entries",NPhiBins_,-3.2,3.2);
        
        jetEt = store_->book1D("L2TauEt","L2 #tau E_{t};L2 selected Jet E_{T};entries",NPtBins_,0,EtMax_);
        jetEta = store_->book1D("L2TauEta","L2 #tau #eta;L2 selected Jet #eta;entries",NEtaBins_,-2.5,2.5);
        jetPhi = store_->book1D("L2TauPhi","L2 #tau #phi;L2 selected Jet #phi;entries",NPhiBins_,-3.2,3.2);
        jetEtRes = store_->book1D("L2TauEtResol","L2 #tau E_{t} resolution;L2 selected Jet #phi;entries",40,-2,2);
        
        isoJetEt = store_->book1D("L2IsoTauEt","L2 isolated #tau E_{t};L2 isolated Jet E_{T};entries",NPtBins_,0,EtMax_);
        isoJetEta = store_->book1D("L2IsoTauEta","L2 isolated #tau #eta;L2 isolated Jet #eta;entries",NEtaBins_,-2.5,2.5);
        isoJetPhi = store_->book1D("L2IsoTauPhi","L2 isolated #tau #phi;L2 isolated Jet #phi;entries",NPhiBins_,-3.2,3.2);
        
        ecalIsolEt = store_->book1D("L2EcalIsolation","ECAL Isolation;L2 ECAL isolation E_{T};entries",40,0,20);
        hcalIsolEt = store_->book1D("L2HcalIsolation","HCAL Isolation;L2 HCAL isolation E_{T};entries",40,0,20);
        
        seedHcalEt = store_->book1D("L2HighestHCALCluster","Highest HCAL Cluster;HCAL seed E_{T};entries",40,0,80);
        seedEcalEt = store_->book1D("L2HighestECALCluster","Highest ECAL Cluster;ECAL seed E_{T};entries",25,0,50);
        
        nEcalClusters = store_->book1D("L2NEcalClusters","Nucmber of ECAL Clusters;n. of ECAL Clusters;entries",20,0,20);
        ecalClusterEtaRMS = store_->book1D("L2EcalEtaRMS","ECAL Cluster #eta RMS;ECAL cluster #eta RMS;entries",15,0,0.05);
        ecalClusterPhiRMS = store_->book1D("L2EcalPhiRMS","ECAL Cluster #phi RMS;ECAL cluster #phi RMS;entries",30,0,0.1);
        ecalClusterDeltaRRMS = store_->book1D("L2EcalDeltaRRMS","ECAL Cluster #DeltaR RMS;ECAL cluster #DeltaR RMS;entries",30,0,0.1);
        
        nHcalClusters = store_->book1D("L2NHcalClusters","Nucmber of HCAL Clusters;n. of ECAL Clusters;entries",20,0,20);
        hcalClusterEtaRMS = store_->book1D("L2HcalEtaRMS","HCAL Cluster #eta RMS;HCAL cluster #eta RMS;entries",15,0,0.05);
        hcalClusterPhiRMS = store_->book1D("L2HcalPhiRMS","HCAL Cluster #phi RMS;HCAL cluster #phi RMS;entries",30,0,0.1);
        hcalClusterDeltaRRMS = store_->book1D("L2HcalDeltaRRMS","HCAL Cluster #DeltaR RMS;HCAL cluster #DeltaR RMS;entries",30,0,0.1);
        
        
        store_->setCurrentFolder(triggerTag()+"/EfficiencyHelpers");
        store_->removeContents();
        
        recoEtEffNum = store_->book1D("L2RecoTauEtEffNum","Efficiency vs E_{t}(Numerator)",NPtBins_,0,EtMax_);
        recoEtEffNum->getTH1F()->Sumw2();
        
        recoEtEffDenom = store_->book1D("L2RecoTauEtEffDenom","Efficiency vs E_{t}(Denominator)",NPtBins_,0,EtMax_);
        recoEtEffDenom->getTH1F()->Sumw2();
        
        recoEtaEffNum = store_->book1D("L2RecoTauEtaEffNum","Efficiency vs #eta (Numerator)",NEtaBins_,-2.5,2.5);
        recoEtaEffNum->getTH1F()->Sumw2();
        
        recoEtaEffDenom = store_->book1D("L2RecoTauEtaEffDenom","Efficiency vs #eta(Denominator)",NEtaBins_,-2.5,2.5);
        recoEtaEffDenom->getTH1F()->Sumw2();
        
        recoPhiEffNum = store_->book1D("L2RecoTauPhiEffNum","Efficiency vs #phi (Numerator)",NPhiBins_,-3.2,3.2);
        recoPhiEffNum->getTH1F()->Sumw2();
        
        recoPhiEffDenom = store_->book1D("L2RecoTauPhiEffDenom","Efficiency vs #phi(Denominator)",NPhiBins_,-3.2,3.2);
        recoPhiEffDenom->getTH1F()->Sumw2();
        
        isoEtEffNum = store_->book1D("L2IsoTauEtEffNum","Efficiency vs E_{t}(Numerator)",NPtBins_,0,EtMax_);
        isoEtEffNum->getTH1F()->Sumw2();
        
        isoEtEffDenom = store_->book1D("L2IsoTauEtEffDenom","Efficiency vs E_{t}(Denominator)",NPtBins_,0,EtMax_);
        isoEtEffDenom->getTH1F()->Sumw2();
        
        isoEtaEffNum = store_->book1D("L2IsoTauEtaEffNum","Efficiency vs #eta (Numerator)",NEtaBins_,-2.5,2.5);
        isoEtaEffNum->getTH1F()->Sumw2();
        
        isoEtaEffDenom = store_->book1D("L2IsoTauEtaEffDenom","Efficiency vs #eta(Denominator)",NEtaBins_,-2.5,2.5);
        isoEtaEffDenom->getTH1F()->Sumw2();
        
        isoPhiEffNum = store_->book1D("L2IsoTauPhiEffNum","Efficiency vs #phi (Numerator)",NPhiBins_,-3.2,3.2);
        isoPhiEffNum->getTH1F()->Sumw2();
        
        isoPhiEffDenom = store_->book1D("L2IsoTauPhiEffDenom","Efficiency vs #phi(Denominator)",NPhiBins_,-3.2,3.2);
        isoPhiEffDenom->getTH1F()->Sumw2();
    }
}

HLTTauDQMCaloPlotter::~HLTTauDQMCaloPlotter() {
}

void HLTTauDQMCaloPlotter::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup, const std::map<int,LVColl>& McInfo ) {
    edm::Handle<reco::L2TauInfoAssociation> l2TauInfoAssoc;
    edm::Handle<reco::CaloJetCollection> l2Isolated;
    edm::Handle<reco::CaloJetCollection> l2Regional;
    reco::CaloJetCollection l2RegionalJets;
    
    //Merge the L2 Regional Collections
    reco::CaloJetCollection l2MergedJets;
    
    for ( unsigned int j = 0; j < l2preJets_.size(); ++j ) {
        bool gotPreJets = iEvent.getByLabel(l2preJets_[j],l2Regional) && l2Regional.isValid();
        
        if (gotPreJets) {
            if ( !l2Regional.failedToGet() ) {
                for ( unsigned int i = 0; i < l2Regional->size(); ++i ) { 
                    l2MergedJets.push_back(l2Regional->at(i));
                }
            }
        }
    }
    
    //Sort
    SorterByPt sorter;
    std::sort(l2MergedJets.begin(),l2MergedJets.end(),sorter);
    
    //Remove Collinear Jets
    reco::CaloJetCollection l2CleanJets;
    while (l2MergedJets.size() > 0 ) {
        l2CleanJets.push_back(l2MergedJets.at(0));
        reco::CaloJetCollection tmp;
        
        for ( unsigned int i = 1; i < l2MergedJets.size(); ++i ) {
            double DR = ROOT::Math::VectorUtil::DeltaR( l2MergedJets.at(0).p4(), l2MergedJets.at(i).p4() );
            if ( DR > 0.1 ) tmp.push_back(l2MergedJets.at(i));
        }
        
        l2MergedJets.swap(tmp);
        tmp.clear();
    }
    
    //Tau reference
    std::map<int,LVColl>::const_iterator iref;
    iref = McInfo.find(15);
    
    //Now fill the regional jet plots by ref if you do ref to avoid double counting!
    if ( doRef_ ) {
        if ( iref != McInfo.end() ) {
            for ( LVColl::const_iterator iter = iref->second.begin(); iter != iref->second.end(); ++iter ) {
                std::pair<bool,reco::CaloJet> m = inverseMatch(*iter,l2CleanJets);
                if ( m.first ) {
                    preJetEt->Fill(m.second.pt());
                    preJetEta->Fill(m.second.eta());
                    preJetPhi->Fill(m.second.phi());
                    recoEtEffDenom->Fill(iter->pt());
                    recoEtaEffDenom->Fill(iter->eta());
                    recoPhiEffDenom->Fill(iter->phi());
                    l2RegionalJets.push_back(m.second);
                }
            }
        }
    } else {
        for ( unsigned int i = 0; i < l2CleanJets.size(); ++i ) {
            reco::CaloJet jet = l2CleanJets.at(i);
            preJetEt->Fill(jet.pt());
            preJetEta->Fill(jet.eta());
            preJetPhi->Fill(jet.phi());
            recoEtEffDenom->Fill(jet.pt());
            recoEtaEffDenom->Fill(jet.eta());
            recoPhiEffDenom->Fill(jet.phi());
            l2RegionalJets.push_back(jet);
        }
    }
    
    bool gotL2 = iEvent.getByLabel(l2TauInfoAssoc_,l2TauInfoAssoc) && l2TauInfoAssoc.isValid();
    
    //If the collection exists do work    
    if ( gotL2 && l2TauInfoAssoc->size() > 0 ) {
        for ( reco::L2TauInfoAssociation::const_iterator p = l2TauInfoAssoc->begin(); p != l2TauInfoAssoc->end(); ++p ) {
            //Retrieve The L2TauIsolationInfo class from the AssociationMap
            const reco::L2TauIsolationInfo l2info = p->val;
            
            //Retrieve the Jet From the AssociationMap
            const reco::Jet& jet =*(p->key);
            
            std::pair<bool,LV> m(false,LV());
            if ( iref != McInfo.end() ) m = match(jet.p4(),iref->second,matchDeltaRMC_);
            
            if ( (doRef_ && m.first) || (!doRef_) ) {
                ecalIsolEt->Fill(l2info.ecalIsolEt());
                hcalIsolEt->Fill(l2info.hcalIsolEt());
                seedEcalEt->Fill(l2info.seedEcalHitEt());
                seedHcalEt->Fill(l2info.seedHcalHitEt());
                
                nEcalClusters->Fill(l2info.nEcalHits());
                ecalClusterEtaRMS->Fill(l2info.ecalClusterShape().at(0));
                ecalClusterPhiRMS->Fill(l2info.ecalClusterShape().at(1));
                ecalClusterDeltaRRMS->Fill(l2info.ecalClusterShape().at(2));
                
                nHcalClusters->Fill(l2info.nHcalHits());
                hcalClusterEtaRMS->Fill(l2info.hcalClusterShape().at(0));
                hcalClusterPhiRMS->Fill(l2info.hcalClusterShape().at(1));
                hcalClusterDeltaRRMS->Fill(l2info.hcalClusterShape().at(2));
                
                jetEt->Fill(jet.et());
                jetEta->Fill(jet.eta());
                jetPhi->Fill(jet.phi());
                
                LV refLV;
                if ( doRef_ ) {
                    refLV = m.second;
                } else {
                    refLV = jet.p4();
                }
                if ( doRef_ ) {
                    jetEtRes->Fill((jet.pt()-refLV.pt())/refLV.pt());
                }
                if ( matchJet(jet,l2RegionalJets) ) {
                    recoEtEffNum->Fill(refLV.pt());
                    recoEtaEffNum->Fill(refLV.eta());
                    recoPhiEffNum->Fill(refLV.phi());
                }
                
                isoEtEffDenom->Fill(refLV.pt());
                isoEtaEffDenom->Fill(refLV.eta());
                isoPhiEffDenom->Fill(refLV.phi());
                
                bool gotIsoL2 = iEvent.getByLabel(l2Isolated_,l2Isolated) && l2Isolated.isValid();
                
                if ( gotIsoL2 ) {
                    if ( matchJet(jet,*l2Isolated) ) { 
                        isoJetEt->Fill(jet.et());
                        isoJetEta->Fill(jet.eta());
                        isoJetPhi->Fill(jet.phi());
                        
                        isoEtEffNum->Fill(refLV.pt());
                        isoEtaEffNum->Fill(refLV.eta());
                        isoPhiEffNum->Fill(refLV.phi());
                    }
                }
            }
        } 
    }
}

std::pair<bool,reco::CaloJet> HLTTauDQMCaloPlotter::inverseMatch( const LV& jet, const reco::CaloJetCollection& jets ) {
    //Loop on the collection and see if your tau jet is matched to one there
    //MATCH THE nearest energy jet in the delta R we want
    
    bool matched = false;
    reco::CaloJet mjet;
    double distance = 100000;
    for ( reco::CaloJetCollection::const_iterator it = jets.begin(); it != jets.end(); ++it ) {
        if ( ROOT::Math::VectorUtil::DeltaR(it->p4(),jet) < matchDeltaRMC_ ) {
            matched = true;
            double delta = fabs(jet.pt()-it->pt());
            if (delta < distance) {
                distance = delta;
                mjet = *it;
            }
        }
    }
    
    std::pair<bool,reco::CaloJet> p = std::make_pair(matched,mjet);
    return p;
}

bool HLTTauDQMCaloPlotter::matchJet( const reco::Jet& jet, const reco::CaloJetCollection& McInfo ) {
    //Loop on the collection and see if your tau jet is matched to one there
    //Also find the nearest matched MC particle to your jet (to be complete)
    
    bool matched = false;
    
    for ( reco::CaloJetCollection::const_iterator it = McInfo.begin(); it != McInfo.end(); ++it ) {
        if ( jet.p4() == it->p4() ) {
            matched = true;
            break;
        }
    }
    return matched;
}
