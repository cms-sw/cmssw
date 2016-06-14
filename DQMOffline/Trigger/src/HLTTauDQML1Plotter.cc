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
  l1stage2Taus_     = ps.getUntrackedParameter<edm::InputTag>("L1Taus");
  l1stage2TausToken_= cc.consumes<l1t::TauBxCollection>(l1stage2Taus_);
 
  l1stage2Sums_     = ps.getUntrackedParameter<edm::InputTag>("L1ETM");
  l1stage2SumsToken_= cc.consumes<l1t::EtSumBxCollection>(l1stage2Sums_);
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
            
    iBooker.setCurrentFolder(triggerTag()+"/helpers");
            
    l1tauEtEffNum_ = iBooker.book1D("L1TauEtEffNum","L1 #tau E_{T} Efficiency;Ref #tau E_{T};entries",binsEt_,0,maxPt_);
    l1tauHighEtEffNum_ = iBooker.book1D("L1TauHighEtEffNum","L1 #tau E_{T} Efficiency (high E_{T});Ref #tau E_{T};entries",binsEt_,0,maxHighPt_);
            
    l1tauEtEffDenom_ = iBooker.book1D("L1TauEtEffDenom","L1 #tau E_{T} Denominator;Ref #tau E_{T};entries",binsEt_,0,maxPt_);
    l1tauHighEtEffDenom_ = iBooker.book1D("L1TauHighEtEffDenom","L1 #tau E_{T} Denominator (high E_{T});Ref #tau E_{T};Efficiency",binsEt_,0,maxHighPt_);
            
    l1tauEtaEffNum_ = iBooker.book1D("L1TauEtaEffNum","L1 #tau #eta Efficiency;Ref #tau #eta;entries",binsEta_,-maxEta_,maxEta_);
    l1tauEtaEffDenom_ = iBooker.book1D("L1TauEtaEffDenom","L1 #tau #eta Denominator;Ref #tau #eta;entries",binsEta_,-maxEta_,maxEta_);
            
    l1tauPhiEffNum_ = iBooker.book1D("L1TauPhiEffNum","L1 #tau #phi Efficiency;Ref #tau #phi;entries",binsPhi_,minPhi,maxPhi);
    l1tauPhiEffDenom_ = iBooker.book1D("L1TauPhiEffDenom","L1 #tau #phi Denominator;Ref #tau #phi;Efficiency",binsPhi_,minPhi,maxPhi);

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
            l1tauHighEtEffDenom_->Fill(iter->pt());
            
            l1tauEtaEffDenom_->Fill(iter->eta());
            
            l1tauPhiEffDenom_->Fill(iter->phi());
        }
	if(refC.met.size() > 0) l1etmEtEffDenom_->Fill(refC.met[0].pt());
    }

    //Analyze L1 Objects (Tau+Jets)
    edm::Handle<l1t::TauBxCollection> taus;
    iEvent.getByToken(l1stage2TausToken_, taus);  
            
    edm::Handle<l1t::EtSumBxCollection> sums;
    iEvent.getByToken(l1stage2SumsToken_, sums);

    LVColl pathTaus;
    
    //Set Variables for the threshold plot
    LVColl l1taus;
    LVColl l1met;

    if(taus.isValid()) {
      for(l1t::TauBxCollection::const_iterator i = taus->begin(); i != taus->end(); ++i) {
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
      edm::LogWarning("HLTTauDQMOffline") << "HLTTauDQML1Plotter::analyze: unable to read L1 tau collection " << l1stage2Taus_.encode();
    }

    if(sums.isValid() && sums.product()->size() > 0) {
      if(!doRefAnalysis_) {
        for (int ibx = sums->getFirstBX(); ibx <= sums->getLastBX(); ++ibx) {
          for (l1t::EtSumBxCollection::const_iterator it=sums->begin(ibx); it!=sums->end(ibx); it++) {
            int type = static_cast<int>( it->getType() );
            if(type == l1t::EtSum::EtSumType::kMissingEt) l1etmEt_->Fill(it->et());
          }
        }
      }
    }
    else {
      edm::LogWarning("HLTTauDQMOffline") << "HLTTauDQML1Plotter::analyze: unable to read L1 met collection " << l1stage2Sums_.encode();
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

        if(sums.isValid() && sums.product()->size() > 0) {
          for (int ibx = sums->getFirstBX(); ibx <= sums->getLastBX(); ++ibx) {
            for (l1t::EtSumBxCollection::const_iterator it=sums->begin(ibx); it!=sums->end(ibx); it++) {
              int type = static_cast<int>( it->getType() );
              if(type == l1t::EtSum::EtSumType::kMissingEt) {
                l1etmEt_->Fill(it->et());
                l1etmPhi_->Fill(it->phi());
        
                if( it->et() > l1ETMMin_){
                  l1etmEtEffNum_->Fill(it->et());
                }
              }
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
