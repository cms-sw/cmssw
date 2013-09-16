#include "DQMOffline/Trigger/interface/HLTTauDQML1Plotter.h"

#include "FWCore/Framework/interface/Event.h"

#include<cstring>

HLTTauDQML1Plotter::HLTTauDQML1Plotter(const edm::ParameterSet& ps, edm::ConsumesCollector&& cc, int etbins, int etabins, int phibins, double maxpt, double maxhighpt, bool ref, double dr, const std::string& dqmBaseFolder):
  HLTTauDQMPlotter(ps, dqmBaseFolder),
  doRefAnalysis_(ref),
  matchDeltaR_(dr),
  maxPt_(maxpt),
  maxHighPt_(maxhighpt),
  binsEt_(etbins),
  binsEta_(etabins),
  binsPhi_(phibins)
{
  if(!configValid_)
    return;

  //Process PSet
  try {
    l1ExtraTaus_      = ps.getUntrackedParameter<edm::InputTag>("L1Taus");
    l1ExtraTausToken_ = cc.consumes<l1extra::L1JetParticleCollection>(l1ExtraTaus_);
    l1ExtraJets_      = ps.getUntrackedParameter<edm::InputTag>("L1Jets");
    l1ExtraJetsToken_ = cc.consumes<l1extra::L1JetParticleCollection>(l1ExtraJets_);
    l1JetMinEt_       = ps.getUntrackedParameter<double>("L1JetMinEt");
  } catch ( cms::Exception &e ) {
    edm::LogWarning("HLTTauDQMOffline") << "HLTTauDQML1Plotter::HLTTauDQML1Plotter: " << e.what();
    configValid_ = false;
    return;
  }
  configValid_ = true;
}

void HLTTauDQML1Plotter::beginRun() {
  if(!configValid_)
    return;

  edm::Service<DQMStore> store;
  if (store.isAvailable()) {
        // The L1 phi plot is asymmetric around 0 because of the discrete nature of L1 phi
        constexpr float pi = 3.1416;
        constexpr float phiShift = pi/18; // half of 2pi/18 bin
        constexpr float minPhi = -pi+phiShift;
        constexpr float maxPhi = pi+phiShift;

        constexpr int BUFMAX = 256;
        char buffer[BUFMAX] = "";

        //Create the histograms
        store->setCurrentFolder(triggerTag());
        store->removeContents();
        
        l1tauEt_ = store->book1D("L1TauEt","L1 #tau E_{T};L1 #tau E_{T};entries",binsEt_,0,maxPt_);
        l1tauEta_ = store->book1D("L1TauEta","L1 #tau #eta;L1 #tau #eta;entries",binsEta_,-2.5,2.5);
        l1tauPhi_ = store->book1D("L1TauPhi","L1 #tau #phi;L1 #tau #phi;entries",binsPhi_,minPhi,maxPhi);
        
        l1jetEt_ = store->book1D("L1JetEt","L1 central jet E_{T};L1 jet E_{T};entries",binsEt_,0,maxPt_);
        snprintf(buffer, BUFMAX, "L1 central jet #eta (E_{T} > %.1f);L1 jet #eta;entries", l1JetMinEt_);
        l1jetEta_ = store->book1D("L1JetEta", buffer, binsEta_, -2.5, 2.5);
        snprintf(buffer, BUFMAX, "L1 central jet #phi (E_{T} > %.1f);L1 jet #phi;entries", l1JetMinEt_);
        l1jetPhi_ = store->book1D("L1JetPhi", buffer, binsPhi_, minPhi, maxPhi);
        
        firstTauEt_ = store->book1D("L1LeadTauEt","L1 leading #tau E_{T};L1 #tau E_{T};entries",binsEt_,0,maxPt_);
        firstTauEta_ = store->book1D("L1LeadTauEta", "L1 leading #tau #eta;L1 #tau #eta;entries",binsEta_,-2.5,2.5);
        firstTauPhi_ = store->book1D("L1LeadTauPhi","L1 leading #tau #phi;L1 #tau #phi;entries",binsPhi_,minPhi,maxPhi);
        // firstTauEt_->getTH1F()->Sumw2(); // why? because of L1(Double|Single)TauEff (now removed)
        
        secondTauEt_ = store->book1D("L1SecondTauEt","L1 second-leading #tau E_{T};L1 #tau E_{T};entries",binsEt_,0,maxPt_);
        secondTauEta_ = store->book1D("L1SecondTauEta","L1 second-leading #tau #eta;L1 #tau #eta;entries",binsEta_,-2.5,2.5);
        secondTauPhi_ = store->book1D("L1SecondTauPhi","L1 second-leading #tau #phi;L1 #tau #phi;entries",binsPhi_,minPhi,maxPhi);
        // secondTauEt_->getTH1F()->Sumw2(); // why? because of L1(Double|Single)TauEff (now removed)
        
        if (doRefAnalysis_) {
            l1tauEtRes_ = store->book1D("L1TauEtResol","L1 #tau E_{T} resolution;[L1 #tau E_{T}-Ref #tau E_{T}]/Ref #tau E_{T};entries",60,-1,4);
            snprintf(buffer, BUFMAX, "L1 central jet E_{T} resolution (E_{T} > %.1f);[L1 jet E_{T}-Ref #tau E_{T}]/Ref #tau E_{T};entries", l1JetMinEt_);
            l1jetEtRes_ = store->book1D("L1JetEtResol", buffer, 60, -1, 4);
            
            store->setCurrentFolder(triggerTag()+"/EfficiencyHelpers");
            store->removeContents();
            
            l1tauEtEffNum_ = store->book1D("L1TauEtEffNum","L1 #tau E_{T} Efficiency;Ref #tau E_{T};entries",binsEt_,0,maxPt_);
            l1tauEtEffNum_->getTH1F()->Sumw2();
            l1tauHighEtEffNum_ = store->book1D("L1TauHighEtEffNum","L1 #tau E_{T} Efficiency (high E_{T});Ref #tau E_{T};entries",binsEt_,0,maxHighPt_);
            l1tauHighEtEffNum_->getTH1F()->Sumw2();
            
            l1tauEtEffDenom_ = store->book1D("L1TauEtEffDenom","L1 #tau E_{T} Denominator;Ref #tau E_{T};entries",binsEt_,0,maxPt_);
            l1tauEtEffDenom_->getTH1F()->Sumw2();
            l1tauHighEtEffDenom_ = store->book1D("L1TauHighEtEffDenom","L1 #tau E_{T} Denominator (high E_{T});Ref #tau E_{T};entries",binsEt_,0,maxHighPt_);
            l1tauHighEtEffDenom_->getTH1F()->Sumw2();
            
            l1tauEtaEffNum_ = store->book1D("L1TauEtaEffNum","L1 #tau #eta Efficiency;Ref #tau #eta;entries",binsEta_,-2.5,2.5);
            l1tauEtaEffNum_->getTH1F()->Sumw2();
            
            l1tauEtaEffDenom_ = store->book1D("L1TauEtaEffDenom","L1 #tau #eta Denominator;Ref #tau #eta;entries",binsEta_,-2.5,2.5);
            l1tauEtaEffDenom_->getTH1F()->Sumw2();
            
            l1tauPhiEffNum_ = store->book1D("L1TauPhiEffNum","L1 #tau #phi Efficiency;Ref #tau #phi;entries",binsPhi_,minPhi,maxPhi);
            l1tauPhiEffNum_->getTH1F()->Sumw2();
            
            l1tauPhiEffDenom_ = store->book1D("L1TauPhiEffDenom","L1 #tau #phi Denominator;Ref #tau #phi;entries",binsPhi_,minPhi,maxPhi);
            l1tauPhiEffDenom_->getTH1F()->Sumw2();
            
            l1jetEtEffNum_ = store->book1D("L1JetEtEffNum","L1 central jet E_{T} Efficiency;Ref #tau E_{T};entries",binsEt_,0,maxPt_);
            l1jetEtEffNum_->getTH1F()->Sumw2();
            l1jetHighEtEffNum_ = store->book1D("L1JetHighEtEffNum","L1 central jet E_{T} Efficiency (high E_{T});Ref #tau E_{T};entries",binsEt_,0,maxHighPt_);
            l1jetHighEtEffNum_->getTH1F()->Sumw2();
            
            l1jetEtEffDenom_ = store->book1D("L1JetEtEffDenom","L1 central jet E_{T} Denominator;Ref #tau E_{T};entries",binsEt_,0,maxPt_);
            l1jetEtEffDenom_->getTH1F()->Sumw2();
            l1jetHighEtEffDenom_ = store->book1D("L1JetHighEtEffDenom","L1 central jet E_{T} Denominator (high E_{T});Ref #tau E_{T};entries",binsEt_,0,maxHighPt_);
            l1jetHighEtEffDenom_->getTH1F()->Sumw2();
            
            snprintf(buffer, BUFMAX, "L1 central jet #eta Efficiency (E_{T} > %.1f);Ref #tau #eta;entries", l1JetMinEt_);
            l1jetEtaEffNum_ = store->book1D("L1JetEtaEffNum", buffer, binsEta_, -2.5, 2.5);
            l1jetEtaEffNum_->getTH1F()->Sumw2();
            
            snprintf(buffer, BUFMAX, "L1 central jet #eta Denominator (E_{T} > %.1f);Ref #tau #eta;entries", l1JetMinEt_);
            l1jetEtaEffDenom_ = store->book1D("L1JetEtaEffDenom", buffer, binsEta_, -2.5, 2.5);
            l1jetEtaEffDenom_->getTH1F()->Sumw2();
            
            snprintf(buffer, BUFMAX, "L1 central jet #phi Efficiency (E_{T} > %.1f);Ref #tau #eta;entries", l1JetMinEt_);
            l1jetPhiEffNum_ = store->book1D("L1JetPhiEffNum", buffer, binsPhi_, minPhi, maxPhi);
            l1jetPhiEffNum_->getTH1F()->Sumw2();
            
            snprintf(buffer, BUFMAX, "L1 central jet #phi Efficiency (E_{T} > %.1f);Ref #tau #eta;entries", l1JetMinEt_);
            l1jetPhiEffDenom_ = store->book1D("L1JetPhiEffDenom", buffer, binsPhi_, minPhi, maxPhi);
            l1jetPhiEffDenom_->getTH1F()->Sumw2();
        }
        runValid_ = true;
  }
  else {
    runValid_ = false;
  }
}


HLTTauDQML1Plotter::~HLTTauDQML1Plotter() {
}

//
// member functions
//

void HLTTauDQML1Plotter::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup, const HLTTauDQMOfflineObjects& refC ) {
    if ( doRefAnalysis_ ) {
        //Tau reference
        for ( LVColl::const_iterator iter = refC.taus.begin(); iter != refC.taus.end(); ++iter ) {
            l1tauEtEffDenom_->Fill(iter->pt());
            l1jetEtEffDenom_->Fill(iter->pt());
            l1tauHighEtEffDenom_->Fill(iter->pt());
            l1jetHighEtEffDenom_->Fill(iter->pt());
            
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
      edm::LogWarning("HLTTauDQMOffline") << "HLTTauDQML1Plotter::analyze: unable to read L1 tau collection " << l1ExtraTaus_.encode();
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
      edm::LogWarning("HLTTauDQMOffline") << "HLTTauDQML1Plotter::analyze: unable to read L1 jet collection " << l1ExtraJets_.encode();
    }
    
    //Now do the efficiency matching
    if ( doRefAnalysis_ ) {
        for ( LVColl::const_iterator i = refC.taus.begin(); i != refC.taus.end(); ++i ) {
            std::pair<bool,LV> m = match(*i,l1taus,matchDeltaR_);
            if ( m.first ) {
                l1tauEt_->Fill(m.second.pt());
                l1tauEta_->Fill(m.second.eta());
                l1tauPhi_->Fill(m.second.phi());

                l1tauEtEffNum_->Fill(i->pt());
                l1tauHighEtEffNum_->Fill(i->pt());
                l1tauEtaEffNum_->Fill(i->eta());
                l1tauPhiEffNum_->Fill(i->phi());

                l1tauEtRes_->Fill((m.second.pt()-i->pt())/i->pt());

                pathTaus.push_back(m.second);
            }
        }
        
        for ( LVColl::const_iterator i = refC.taus.begin(); i != refC.taus.end(); ++i ) {
            std::pair<bool,LV> m = match(*i,l1jets,matchDeltaR_);
            if ( m.first ) {
                l1jetEt_->Fill(m.second.pt());
                if(m.second.pt() >= l1JetMinEt_) {
                  l1jetEta_->Fill(m.second.eta());
                  l1jetPhi_->Fill(m.second.phi());

                  l1jetEtEffNum_->Fill(i->pt());
                  l1jetHighEtEffNum_->Fill(i->pt());
                  l1jetEtaEffNum_->Fill(i->eta());
                  l1jetPhiEffNum_->Fill(i->phi());

                  l1jetEtRes_->Fill((m.second.pt()-i->pt())/i->pt());

                  pathTaus.push_back(m.second);
                }
            }
        }
    }
    
    
    //Fill the Threshold Monitoring
    if(pathTaus.size() > 1) std::sort(pathTaus.begin(), pathTaus.end(), [](const LV& a, const LV& b) { return a.pt() > b.pt(); });
    
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
