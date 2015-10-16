#include "DQMOffline/Trigger/interface/HLTTauDQML1Plotter.h"

#include "FWCore/Framework/interface/Event.h"

#include<cstring>

namespace {
  double getMaxEta(int binsEta, double widthEta) {
    if(widthEta <= 0.0) {
      edm::LogWarning("HLTTauDQMOffline") << "HLTTauDQML1Plotter::HLTTauDQML1Plotter: EtaHistoBinWidth = " << widthEta << " <= 0, using default value 0.348 instead";
      widthEta = 0.348;
    }
    return binsEta/2*widthEta;
  }
}

HLTTauDQML1Plotter::HLTTauDQML1Plotter(const edm::ParameterSet& ps, edm::ConsumesCollector&& cc, int phibins, double maxpt, double maxhighpt, bool ref, double dr, const std::string& dqmBaseFolder):
  HLTTauDQMPlotter(ps, dqmBaseFolder),
  doRefAnalysis_(ref),
  matchDeltaR_(dr),
  maxPt_(maxpt),
  maxHighPt_(maxhighpt),
  binsEt_(ps.getUntrackedParameter<int>("EtHistoBins", 25)),
  binsEta_(ps.getUntrackedParameter<int>("EtaHistoBins", 14)),
  binsPhi_(phibins),
  maxEta_(getMaxEta(binsEta_, ps.getUntrackedParameter<double>("EtaHistoBinWidth", 0.348)))
{
  if(!configValid_)
    return;

  //Process PSet
  l1ExtraTaus_      = ps.getUntrackedParameter<edm::InputTag>("L1Taus");
  l1ExtraTausToken_ = cc.consumes<l1extra::L1JetParticleCollection>(l1ExtraTaus_);
  l1ExtraIsoTaus_      = ps.getUntrackedParameter<edm::InputTag>("L1IsoTaus");
  l1ExtraIsoTausToken_ = cc.consumes<l1extra::L1JetParticleCollection>(l1ExtraIsoTaus_);
  l1ExtraJets_      = ps.getUntrackedParameter<edm::InputTag>("L1Jets");
  l1ExtraJetsToken_ = cc.consumes<l1extra::L1JetParticleCollection>(l1ExtraJets_);
  l1ExtraMET_       = ps.getUntrackedParameter<edm::InputTag>("L1ETM");
  l1ExtraMETToken_  = cc.consumes<l1extra::L1EtMissParticleCollection>(l1ExtraMET_);
  l1JetMinEt_       = ps.getUntrackedParameter<double>("L1JetMinEt");
  l1ETMMin_         = ps.getUntrackedParameter<double>("L1ETMMin");
  configValid_ = true;
}

void HLTTauDQML1Plotter::bookHistograms(DQMStore::IBooker &iBooker) {
  if(!configValid_)
    return;

  // The L1 phi plot is asymmetric around 0 because of the discrete nature of L1 phi
  constexpr float pi = 3.1416f;
  constexpr float phiShift = pi/18; // half of 2pi/18 bin
  constexpr float minPhi = -pi+phiShift;
  constexpr float maxPhi = pi+phiShift;

  constexpr int BUFMAX = 256;
  char buffer[BUFMAX] = "";

  //Create the histograms
  iBooker.setCurrentFolder(triggerTag());
        
  l1tauEt_ = iBooker.book1D("L1TauEt","L1 #tau E_{T};L1 #tau E_{T};entries",binsEt_,0,maxPt_);
  l1tauEta_ = iBooker.book1D("L1TauEta","L1 #tau #eta;L1 #tau #eta;entries",binsEta_,-maxEta_,maxEta_);
  l1tauPhi_ = iBooker.book1D("L1TauPhi","L1 #tau #phi;L1 #tau #phi;entries",binsPhi_,minPhi,maxPhi);

  l1isotauEt_ = iBooker.book1D("L1IsoTauEt","L1 iso#tau E_{T};L1 iso#tau E_{T};entries",binsEt_,0,maxPt_);
  l1isotauEta_ = iBooker.book1D("L1IsoTauEta","L1 iso#tau #eta;L1 iso#tau #eta;entries",binsEta_,-maxEta_,maxEta_);
  l1isotauPhi_ = iBooker.book1D("L1IsoTauPhi","L1 iso#tau #phi;L1 iso#tau #phi;entries",binsPhi_,minPhi,maxPhi);
        
  l1jetEt_ = iBooker.book1D("L1JetEt","L1 central jet E_{T};L1 jet E_{T};entries",binsEt_,0,maxPt_);
  snprintf(buffer, BUFMAX, "L1 central jet #eta (E_{T} > %.1f);L1 jet #eta;entries", l1JetMinEt_);
  l1jetEta_ = iBooker.book1D("L1JetEta", buffer, binsEta_, -maxEta_, maxEta_);
  snprintf(buffer, BUFMAX, "L1 central jet #phi (E_{T} > %.1f);L1 jet #phi;entries", l1JetMinEt_);
  l1jetPhi_ = iBooker.book1D("L1JetPhi", buffer, binsPhi_, minPhi, maxPhi);

  l1etmEt_  = iBooker.book1D("L1ETM","L1 ETM E_{T};L1 ETM E_{T};entries",binsEt_,0,maxPt_);
  l1etmPhi_ = iBooker.book1D("L1ETMPhi","L1 ETM #phi;L1 ETM #phi;entries",binsPhi_,minPhi,maxPhi);
        
  snprintf(buffer, BUFMAX, "L1 leading (#tau OR central jet E_{T} > %.1f) E_{T};L1 (#tau or central jet) E_{T};entries", l1JetMinEt_);
  firstTauEt_ = iBooker.book1D("L1LeadTauEt", buffer, binsEt_, 0, maxPt_);
  snprintf(buffer, BUFMAX, "L1 leading (#tau OR central jet E_{T} > %.1f) #eta;L1 (#tau or central jet) #eta;entries", l1JetMinEt_);
  firstTauEta_ = iBooker.book1D("L1LeadTauEta", buffer, binsEta_, -maxEta_, maxEta_);
  snprintf(buffer, BUFMAX, "L1 leading (#tau OR central jet E_{T} > %.1f) #phi;L1 (#tau or central jet) #phi;entries", l1JetMinEt_);
  firstTauPhi_ = iBooker.book1D("L1LeadTauPhi", buffer, binsPhi_, minPhi, maxPhi);
        
  snprintf(buffer, BUFMAX, "L1 second-leading (#tau OR central jet E_{T} > %.1f) E_{T};L1 (#tau or central jet) E_{T};entries", l1JetMinEt_);
  secondTauEt_ = iBooker.book1D("L1SecondTauEt", buffer, binsEt_, 0, maxPt_);
  snprintf(buffer, BUFMAX, "L1 second-leading (#tau OR central jet E_{T} > %.1f) #eta;L1 (#tau or central jet) #eta;entries", l1JetMinEt_);
  secondTauEta_ = iBooker.book1D("L1SecondTauEta", buffer, binsEta_, -maxEta_, maxEta_);
  snprintf(buffer, BUFMAX, "L1 second-leading (#tau OR central jet E_{T} > %.1f) #phi;L1 (#tau or central jet) #phi;entries", l1JetMinEt_);
  secondTauPhi_ = iBooker.book1D("L1SecondTauPhi", buffer, binsPhi_, minPhi, maxPhi);
        
  if (doRefAnalysis_) {
    l1tauEtRes_ = iBooker.book1D("L1TauEtResol","L1 #tau E_{T} resolution;[L1 #tau E_{T}-Ref #tau E_{T}]/Ref #tau E_{T};entries",60,-1,4);
    l1isotauEtRes_ = iBooker.book1D("L1IsoTauEtResol","L1 iso#tau E_{T} resolution;[L1 iso#tau E_{T}-Ref iso#tau E_{T}]/Ref iso#tau E_{T};entries",60,-1,4);
    snprintf(buffer, BUFMAX, "L1 central jet E_{T} resolution (E_{T} > %.1f);[L1 jet E_{T}-Ref #tau E_{T}]/Ref #tau E_{T};entries", l1JetMinEt_);
    l1jetEtRes_ = iBooker.book1D("L1JetEtResol", buffer, 60, -1, 4);
            
    iBooker.setCurrentFolder(triggerTag()+"/helpers");
            
    l1tauEtEffNum_ = iBooker.book1D("L1TauEtEffNum","L1 #tau E_{T} Efficiency;Ref #tau E_{T};entries",binsEt_,0,maxPt_);
    l1tauHighEtEffNum_ = iBooker.book1D("L1TauHighEtEffNum","L1 #tau E_{T} Efficiency (high E_{T});Ref #tau E_{T};entries",binsEt_,0,maxHighPt_);
            
    l1tauEtEffDenom_ = iBooker.book1D("L1TauEtEffDenom","L1 #tau E_{T} Denominator;Ref #tau E_{T};entries",binsEt_,0,maxPt_);
    l1tauHighEtEffDenom_ = iBooker.book1D("L1TauHighEtEffDenom","L1 #tau E_{T} Denominator (high E_{T});Ref #tau E_{T};Efficiency",binsEt_,0,maxHighPt_);
            
    l1tauEtaEffNum_ = iBooker.book1D("L1TauEtaEffNum","L1 #tau #eta Efficiency;Ref #tau #eta;entries",binsEta_,-maxEta_,maxEta_);
    l1tauEtaEffDenom_ = iBooker.book1D("L1TauEtaEffDenom","L1 #tau #eta Denominator;Ref #tau #eta;entries",binsEta_,-maxEta_,maxEta_);
            
    l1tauPhiEffNum_ = iBooker.book1D("L1TauPhiEffNum","L1 #tau #phi Efficiency;Ref #tau #phi;entries",binsPhi_,minPhi,maxPhi);
    l1tauPhiEffDenom_ = iBooker.book1D("L1TauPhiEffDenom","L1 #tau #phi Denominator;Ref #tau #phi;Efficiency",binsPhi_,minPhi,maxPhi);

    l1isotauEtEffNum_ = iBooker.book1D("L1IsoTauEtEffNum","L1 iso#tau E_{T} Efficiency;Ref iso#tau E_{T};entries",binsEt_,0,maxPt_);                                                                        
    l1isotauHighEtEffNum_ = iBooker.book1D("L1IsoTauHighEtEffNum","L1 iso#tau E_{T} Efficiency (high E_{T});Ref iso#tau E_{T};entries",binsEt_,0,maxHighPt_);                                               
                                                                                                                                                                                                
    l1isotauEtEffDenom_ = iBooker.book1D("L1IsoTauEtEffDenom","L1 iso#tau E_{T} Denominator;Ref iso#tau E_{T};entries",binsEt_,0,maxPt_);                                                                   
    l1isotauHighEtEffDenom_ = iBooker.book1D("L1IsoTauHighEtEffDenom","L1 iso#tau E_{T} Denominator (high E_{T});Ref iso#tau E_{T};Efficiency",binsEt_,0,maxHighPt_);                                       
                                                                                                                                                                                                
    l1isotauEtaEffNum_ = iBooker.book1D("L1IsoTauEtaEffNum","L1 iso#tau #eta Efficiency;Ref iso#tau #eta;entries",binsEta_,-maxEta_,maxEta_);                                                               
    l1isotauEtaEffDenom_ = iBooker.book1D("L1IsoTauEtaEffDenom","L1 iso#tau #eta Denominator;Ref iso#tau #eta;entries",binsEta_,-maxEta_,maxEta_);                                                          
                                                                                                                                                                                                
    l1isotauPhiEffNum_ = iBooker.book1D("L1IsoTauPhiEffNum","L1 iso#tau #phi Efficiency;Ref iso#tau #phi;entries",binsPhi_,minPhi,maxPhi);                                                                  
    l1isotauPhiEffDenom_ = iBooker.book1D("L1IsoTauPhiEffDenom","L1 iso#tau #phi Denominator;Ref iso#tau #phi;Efficiency",binsPhi_,minPhi,maxPhi);
            
    l1jetEtEffNum_ = iBooker.book1D("L1JetEtEffNum","L1 central jet E_{T} Efficiency;Ref #tau E_{T};entries",binsEt_,0,maxPt_);
    l1jetHighEtEffNum_ = iBooker.book1D("L1JetHighEtEffNum","L1 central jet E_{T} Efficiency (high E_{T});Ref #tau E_{T};entries",binsEt_,0,maxHighPt_);
            
    l1jetEtEffDenom_ = iBooker.book1D("L1JetEtEffDenom","L1 central jet E_{T} Denominator;Ref #tau E_{T};entries",binsEt_,0,maxPt_);
    l1jetHighEtEffDenom_ = iBooker.book1D("L1JetHighEtEffDenom","L1 central jet E_{T} Denominator (high E_{T});Ref #tau E_{T};Efficiency",binsEt_,0,maxHighPt_);
            
    snprintf(buffer, BUFMAX, "L1 central jet #eta Efficiency (E_{T} > %.1f);Ref #tau #eta;entries", l1JetMinEt_);
    l1jetEtaEffNum_ = iBooker.book1D("L1JetEtaEffNum", buffer, binsEta_, -maxEta_, maxEta_);
            
    snprintf(buffer, BUFMAX, "L1 central jet #eta Denominator (E_{T} > %.1f);Ref #tau #eta;Efficiency", l1JetMinEt_);
    l1jetEtaEffDenom_ = iBooker.book1D("L1JetEtaEffDenom", buffer, binsEta_, -maxEta_, maxEta_);
            
    snprintf(buffer, BUFMAX, "L1 central jet #phi Efficiency (E_{T} > %.1f);Ref #tau #phi;entries", l1JetMinEt_);
    l1jetPhiEffNum_ = iBooker.book1D("L1JetPhiEffNum", buffer, binsPhi_, minPhi, maxPhi);
            
    snprintf(buffer, BUFMAX, "L1 central jet #phi Efficiency (E_{T} > %.1f);Ref #tau #phi;Efficiency", l1JetMinEt_);
    l1jetPhiEffDenom_ = iBooker.book1D("L1JetPhiEffDenom", buffer, binsPhi_, minPhi, maxPhi);

    l1etmEtEffNum_ = iBooker.book1D("L1ETMEtEffNum", "L1 ETM Efficiency;Ref MET;entries",binsEt_, 0, maxPt_);
    l1etmEtEffDenom_ = iBooker.book1D("L1ETMEtEffDenom","L1 ETM Denominator;Ref MET;entries",binsEt_,0,maxPt_);
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
            l1isotauEtEffDenom_->Fill(iter->pt());
            l1jetEtEffDenom_->Fill(iter->pt());
            l1tauHighEtEffDenom_->Fill(iter->pt());
            l1isotauHighEtEffDenom_->Fill(iter->pt());
            l1jetHighEtEffDenom_->Fill(iter->pt());
            
            l1tauEtaEffDenom_->Fill(iter->eta());
            l1isotauEtaEffDenom_->Fill(iter->eta());
            l1jetEtaEffDenom_->Fill(iter->eta());
            
            l1tauPhiEffDenom_->Fill(iter->phi());
            l1isotauPhiEffDenom_->Fill(iter->phi());
            l1jetPhiEffDenom_->Fill(iter->phi());
        }
	if(refC.met.size() > 0) l1etmEtEffDenom_->Fill(refC.met[0].pt());
    }
    
    //Analyze L1 Objects (Tau+Jets)
    edm::Handle<l1extra::L1JetParticleCollection> taus;
    edm::Handle<l1extra::L1JetParticleCollection> isotaus;
    edm::Handle<l1extra::L1JetParticleCollection> jets;
    edm::Handle<l1extra::L1EtMissParticleCollection> met;
    iEvent.getByToken(l1ExtraTausToken_, taus);
    iEvent.getByToken(l1ExtraIsoTausToken_, isotaus);
    iEvent.getByToken(l1ExtraJetsToken_, jets);
    iEvent.getByToken(l1ExtraMETToken_, met);
    
    LVColl pathTaus;
    
    //Set Variables for the threshold plot
    LVColl l1taus;
    LVColl l1isotaus;
    LVColl l1jets;
    LVColl l1met;

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

    if(isotaus.isValid()) {
      for(l1extra::L1JetParticleCollection::const_iterator i = isotaus->begin(); i != isotaus->end(); ++i) {
        l1isotaus.push_back(i->p4());
        if(!doRefAnalysis_) {
          l1isotauEt_->Fill(i->et());
          l1isotauEta_->Fill(i->eta());
          l1isotauPhi_->Fill(i->phi());
          pathTaus.push_back(i->p4());
        }
      }
    }
    else {
      edm::LogWarning("HLTTauDQMOffline") << "HLTTauDQML1Plotter::analyze: unable to read L1 isotau collection " << l1ExtraIsoTaus_.encode();
    }

    if(jets.isValid()) {
      for(l1extra::L1JetParticleCollection::const_iterator i = jets->begin(); i != jets->end(); ++i) {
        l1jets.push_back(i->p4());
        if(!doRefAnalysis_) {
          l1jetEt_->Fill(i->et());
          if(i->et() >= l1JetMinEt_) {
            l1jetEta_->Fill(i->eta());
            l1jetPhi_->Fill(i->phi());
            pathTaus.push_back(i->p4());
          }
        }
      }
    }
    else {
      edm::LogWarning("HLTTauDQMOffline") << "HLTTauDQML1Plotter::analyze: unable to read L1 jet collection " << l1ExtraJets_.encode();
    }

    if(met.isValid() && met.product()->size() > 0) {
      if(!doRefAnalysis_) {
	if( met.product()->begin()->et() > l1ETMMin_) l1etmEt_->Fill(met.product()->begin()->et());
      }
    }
    else {
      edm::LogWarning("HLTTauDQMOffline") << "HLTTauDQML1Plotter::analyze: unable to read L1 met collection " << l1ExtraMET_.encode();
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
            std::pair<bool,LV> m = match(*i,l1isotaus,matchDeltaR_);
            if ( m.first ) {
                l1isotauEt_->Fill(m.second.pt());
                l1isotauEta_->Fill(m.second.eta());  
                l1isotauPhi_->Fill(m.second.phi());  
         
                l1isotauEtEffNum_->Fill(i->pt());
                l1isotauHighEtEffNum_->Fill(i->pt());
                l1isotauEtaEffNum_->Fill(i->eta());
                l1isotauPhiEffNum_->Fill(i->phi());
             
                l1isotauEtRes_->Fill((m.second.pt()-i->pt())/i->pt());
    
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

	if(met.isValid() && met.product()->size() > 0) {
          l1etmEt_->Fill(met.product()->begin()->et());
          l1etmPhi_->Fill(met.product()->begin()->phi());

	  if( met.product()->begin()->et() > l1ETMMin_){
            l1etmEtEffNum_->Fill(met.product()->begin()->et());
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
