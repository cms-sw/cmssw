#include "DQMOffline/Trigger/interface/HLTTauDQML1Plotter.h"

namespace {
  struct ComparePt {
    bool operator() (LV l1,LV l2) {
      return l1.pt() > l2.pt();
    }
  };
}

HLTTauDQML1Plotter::HLTTauDQML1Plotter(const edm::ParameterSet& ps, edm::ConsumesCollector&& cc, int etbins, int etabins, int phibins, double maxpt, bool ref, double dr, std::string dqmBaseFolder) {
    //Initialize Plotter
    name_ = "HLTTauDQML1Plotter";
    
    //Process PSet
    try {
        triggerTag_       = ps.getUntrackedParameter<std::string>("DQMFolder");
        triggerTagAlias_  = ps.getUntrackedParameter<std::string>("Alias","");
        l1ExtraTaus_      = ps.getUntrackedParameter<edm::InputTag>("L1Taus");
        l1ExtraTausToken_ = cc.consumes<l1extra::L1JetParticleCollection>(l1ExtraTaus_);
        l1ExtraJets_      = ps.getUntrackedParameter<edm::InputTag>("L1Jets");
        l1ExtraJetsToken_ = cc.consumes<l1extra::L1JetParticleCollection>(l1ExtraJets_);
        doRefAnalysis_    = ref;
        dqmBaseFolder_    = dqmBaseFolder;
        matchDeltaR_      = dr;
        l1JetMinEt_       = ps.getUntrackedParameter<double>("L1JetMinEt");
        maxEt_            = maxpt;
        binsEt_           = etbins;
        binsEta_          = etabins;
        binsPhi_          = phibins;
        validity_         = true;
    } catch ( cms::Exception &e ) {
      edm::LogInfo("HLTTauDQMOffline") << "HLTTauDQML1Plotter::HLTTauDQML1Plotter: " << e.what() << std::endl;
      validity_ = false;
      return;
    }
}

void HLTTauDQML1Plotter::beginRun() {
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
        
        firstTauEt_ = store_->book1D("L1LeadTauEt","L1 lead #tau E_{T};entries",binsEt_,0,maxEt_);
        firstTauEta_ = store_->book1D("L1LeadTauEta","L1 lead #tau #eta;entries",binsEta_,-2.5,2.5);
        firstTauPhi_ = store_->book1D("L1LeadTauPhi","L1 lead #tau #phi;entries",binsPhi_,-3.2,3.2);
        // firstTauEt_->getTH1F()->Sumw2(); // why? because of L1(Double|Single)TauEff (now removed)
        
        secondTauEt_ = store_->book1D("L1SecondTauEt","L1 second #tau E_{T}",binsEt_,0,maxEt_);
        secondTauEta_ = store_->book1D("L1SecondTauEta","L1 second #tau E_{T}",binsEta_,-2.5,2.5);
        secondTauPhi_ = store_->book1D("L1SecondTauPhi","L1 second #tau E_{T}",binsPhi_,-3.2,3.2);
        // secondTauEt_->getTH1F()->Sumw2(); // why? because of L1(Double|Single)TauEff (now removed)
        
        if (doRefAnalysis_) {
            l1tauEtRes_ = store_->book1D("L1TauEtResol","L1 #tau E_{T} resolution;[L1 #tau E_{T}-Ref #tau E_{T}]/Ref #tau E_{T};entries",60,-1,4);
            l1jetEtRes_ = store_->book1D("L1JetEtResol","L1 central jet E_{T} resolution;[L1 #jet E_{T}-Ref #tau E_{T}]/Ref #tau E_{T};entries",60,-1,4);
            
            store_->setCurrentFolder(triggerTag()+"/EfficiencyHelpers");
            store_->removeContents();
            
            l1tauEtEffNum_ = store_->book1D("L1TauEtEffNum","L1 #tau E_{T} Efficiency Numerator;Ref #tau E_{T};entries",binsEt_,0,maxEt_);
            l1tauEtEffNum_->getTH1F()->Sumw2();
            
            l1tauEtEffDenom_ = store_->book1D("L1TauEtEffDenom","L1 #tau E_{T} Denominator;Ref #tau E_{T};entries",binsEt_,0,maxEt_);
            l1tauEtEffDenom_->getTH1F()->Sumw2();
            
            l1tauEtaEffNum_ = store_->book1D("L1TauEtaEffNum","L1 #tau #eta Efficiency;Ref #tau E_{T};entries",binsEta_,-2.5,2.5);
            l1tauEtaEffNum_->getTH1F()->Sumw2();
            
            l1tauEtaEffDenom_ = store_->book1D("L1TauEtaEffDenom","L1 #tau #eta Denominator;Ref #tau E_{T};entries",binsEta_,-2.5,2.5);
            l1tauEtaEffDenom_->getTH1F()->Sumw2();
            
            l1tauPhiEffNum_ = store_->book1D("L1TauPhiEffNum","L1 #tau #phi Efficiency;Ref #tau E_{T};entries",binsPhi_,-3.2,3.2);
            l1tauPhiEffNum_->getTH1F()->Sumw2();
            
            l1tauPhiEffDenom_ = store_->book1D("L1TauPhiEffDenom","L1 #tau #phi Denominator;Ref #tau E_{T};entries",binsPhi_,-3.2,3.2);
            l1tauPhiEffDenom_->getTH1F()->Sumw2();
            
            l1jetEtEffNum_ = store_->book1D("L1JetEtEffNum","L1 jet E_{T} Efficiency;Ref #tau E_{T};entries",binsEt_,0,maxEt_);
            l1jetEtEffNum_->getTH1F()->Sumw2();
            
            l1jetEtEffDenom_ = store_->book1D("L1JetEtEffDenom","L1 jet E_{T} Denominator;Ref #tau E_{T};entries",binsEt_,0,maxEt_);
            l1jetEtEffDenom_->getTH1F()->Sumw2();
            
            l1jetEtaEffNum_ = store_->book1D("L1JetEtaEffNum","L1 jet #eta Efficiency;Ref #tau E_{T};entries",binsEta_,-2.5,2.5);
            l1jetEtaEffNum_->getTH1F()->Sumw2();
            
            l1jetEtaEffDenom_ = store_->book1D("L1JetEtaEffDenom","L1 jet #eta Denominator;Ref #tau E_{T};entries",binsEta_,-2.5,2.5);
            l1jetEtaEffDenom_->getTH1F()->Sumw2();
            
            l1jetPhiEffNum_ = store_->book1D("L1JetPhiEffNum","L1 jet #phi Efficiency;Ref #tau E_{T};entries",binsPhi_,-3.2,3.2);
            l1jetPhiEffNum_->getTH1F()->Sumw2();
            
            l1jetPhiEffDenom_ = store_->book1D("L1JetPhiEffDenom","L1 jet #phi Denominator;Ref #tau E_{T};entries",binsPhi_,-3.2,3.2);
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
  LVColl refTaus;
    
    if ( doRefAnalysis_ ) {
        std::map<int,LVColl>::const_iterator iref;

        //Tau reference
        iref = refC.find(15);
        if ( iref != refC.end() ) refTaus = iref->second;
        
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
    iEvent.getByToken(l1ExtraTausToken_, taus);
    iEvent.getByToken(l1ExtraJetsToken_, jets);
    
    LVColl pathTaus;
    
    //Set Variables for the threshold plot
    LVColl l1taus;
    LVColl l1jets;

    if(taus.isValid()) {
      for(l1extra::L1JetParticleCollection::const_iterator i = taus->begin(); i != taus->end(); ++i) {
        l1taus.push_back(i->p4());
        if(!doRefAnalysis_) {
          l1tauEt_->Fill(i->et());
          l1tauEta_->Fill(i->eta());
          l1tauPhi_->Fill(i->phi());
          pathTaus.push_back(i->p4());
        }
      }
    }
    else {
      edm::LogInfo("HLTTauDQMOffline") << "HLTTauDQML1Plotter::analyze: unable to read L1 tau collection " << l1ExtraTaus_.encode() << std::endl;
    }

    if(jets.isValid()) {
      for(l1extra::L1JetParticleCollection::const_iterator i = jets->begin(); i != jets->end(); ++i) {
        l1jets.push_back(i->p4());
        if(!doRefAnalysis_) {
          l1jetEt_->Fill(i->et());
          if(i->et() >= l1JetMinEt_) {
            l1jetEta_->Fill(i->eta());
            l1jetPhi_->Fill(i->phi());
          }
          pathTaus.push_back(i->p4());
        }
      }
    }
    else {
      edm::LogInfo("HLTTauDQMOffline") << "HLTTauDQML1Plotter::analyze: unable to read L1 jet collection " << l1ExtraJets_.encode() << std::endl;
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
                if(m.second.pt() >= l1JetMinEt_) {
                  l1jetEta_->Fill(m.second.eta());
                  l1jetPhi_->Fill(m.second.phi());

                  l1jetEtEffNum_->Fill(i->pt());
                  l1jetEtaEffNum_->Fill(i->eta());
                  l1jetPhiEffNum_->Fill(i->phi());

                  l1jetEtRes_->Fill((m.second.pt()-i->pt())/i->pt());

                  pathTaus.push_back(m.second);
                }
            }
        }
    }
    
    
    //Fill the Threshold Monitoring
    if(pathTaus.size() > 1) std::sort(pathTaus.begin(),pathTaus.end(),ComparePt());
    
    if ( pathTaus.size() > 0 ) {
        firstTauEt_->Fill(pathTaus[0].pt());
        firstTauEta_->Fill(pathTaus[0].eta());
        firstTauPhi_->Fill(pathTaus[0].phi());
    }
    if ( pathTaus.size() > 1 ) {
        secondTauEt_->Fill(pathTaus[1].pt());
        secondTauEta_->Fill(pathTaus[1].eta());
        secondTauPhi_->Fill(pathTaus[1].phi());
    }
}
