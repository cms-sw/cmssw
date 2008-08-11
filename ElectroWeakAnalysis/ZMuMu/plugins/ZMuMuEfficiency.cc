/* \class ZMuMuEfficiency
 * 
 * author: Pasquale Noli
 * revised by Salvatore di Guida
 *
 * Efficiency of reconstruction tracker and muon Chamber
 *
 */

#include "DataFormats/Common/interface/AssociationVector.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "TH1.h"
#include <vector>

class ZMuMuEfficiency : public edm::EDAnalyzer {
public:
  ZMuMuEfficiency(const edm::ParameterSet& pset);
private:
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual void endJob();

  edm::InputTag zMuTrack_, zMuTrackMatchMap_; 
  edm::InputTag zMuStandAlone_, zMuStandAloneMatchMap_;
  edm::InputTag muons_, muonMatchMap_, muonIso_;
  edm::InputTag tracks_, trackIso_;
  edm::InputTag standAlone_, standAloneIso_;
  double zMassMin_, zMassMax_, ptmin_, etamax_, isomax_;
  size_t nbinsPt_, nbinsEta_;
  reco::CandidateRef globalMuonCandRef_, trackMuonCandRef_, standAloneMuonCandRef_;
  OverlapChecker overlap_;

  //histograms for measuring tracker efficiency
  TH1D *h_etaStandAlone_, *h_etaMuonOverlappedToStandAlone_; 
  TH1D *h_ptStandAlone_, *h_ptMuonOverlappedToStandAlone_; 

  //histograms for measuring standalone efficiency
  TH1D *h_etaTrack_, *h_etaMuonOverlappedToTrack_;
  TH1D *h_ptTrack_, *h_ptMuonOverlappedToTrack_;

  int numberOfMatchedZMuSta_, numberOfMatchedZMuTrack_;
  int numberOfOverlappedStandAlone_, numberOfOverlappedTracks_;
};

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include <iostream>
#include <iterator>
#include <cmath>
using namespace std;
using namespace reco;
using namespace edm;

typedef CandDoubleAssociations IsolationCollection;

ZMuMuEfficiency::ZMuMuEfficiency(const ParameterSet& pset) : 
  zMuTrack_(pset.getParameter<InputTag>("zMuTrack")), 
  zMuTrackMatchMap_(pset.getParameter<InputTag>("zMuTrackMatchMap")), 
  zMuStandAlone_(pset.getParameter<InputTag>("zMuStandAlone")), 
  zMuStandAloneMatchMap_(pset.getParameter<InputTag>("zMuStandAloneMatchMap")), 
  muons_(pset.getParameter<InputTag>("muons")), 
  muonMatchMap_(pset.getParameter<InputTag>("muonMatchMap")), 
  muonIso_(pset.getParameter<InputTag>("muonIso")), 
  tracks_(pset.getParameter<InputTag>("tracks")), 
  trackIso_(pset.getParameter<InputTag>("trackIso")), 
  standAlone_(pset.getParameter<InputTag>("standAlone")), 
  standAloneIso_(pset.getParameter<InputTag>("standAloneIso")), 
  zMassMin_(pset.getUntrackedParameter<double>("zMassMin")), 
  zMassMax_(pset.getUntrackedParameter<double>("zMassMax")), 
  ptmin_(pset.getUntrackedParameter<double>("ptmin")), 
  etamax_(pset.getUntrackedParameter<double>("etamax")),  
  isomax_(pset.getUntrackedParameter<double>("isomax")), 
  nbinsPt_(pset.getUntrackedParameter<size_t>("nbinsPt")), 
  nbinsEta_(pset.getUntrackedParameter<size_t>("nbinsEta")) {
  Service<TFileService> fs;
  TFileDirectory trackEffDir = fs->mkdir("TrackEfficiency");
  h_etaStandAlone_ = trackEffDir.make<TH1D>("StandAloneMuonEta", 
					    "StandAlone #eta for Z -> #mu + standalone", 
					    nbinsEta_, -etamax_, etamax_);
  h_etaMuonOverlappedToStandAlone_ = trackEffDir.make<TH1D>("MuonOverlappedToStandAloneEta", 
							    "Global muon overlapped to standAlone #eta for Z -> #mu + sa", 
							    nbinsEta_, -etamax_, etamax_);
  h_ptStandAlone_ = trackEffDir.make<TH1D>("StandAloneMuonPt", 
					   "StandAlone p_{t} for Z -> #mu + standalone", 
					   nbinsPt_, ptmin_, 200);
  h_ptMuonOverlappedToStandAlone_ = trackEffDir.make<TH1D>("MuonOverlappedToStandAlonePt", 
							   "Global muon overlapped to standAlone p_{t} for Z -> #mu + sa", 
							   nbinsPt_, ptmin_, 200);
  
  
  TFileDirectory standaloneEffDir = fs->mkdir("StandaloneEfficiency");
  h_etaTrack_ = standaloneEffDir.make<TH1D>("TrackMuonEta", 
					    "Track #eta for Z -> #mu + track", 
					    nbinsEta_, -etamax_, etamax_);
  h_etaMuonOverlappedToTrack_ = standaloneEffDir.make<TH1D>("MuonOverlappedToTrackEta", 
							    "Global muon overlapped to track #eta for Z -> #mu + tk", 
							    nbinsEta_, -etamax_, etamax_);
  h_ptTrack_ = standaloneEffDir.make<TH1D>("TrackMuonPt", 
					   "Track p_{t} for Z -> #mu + track", 
					   nbinsPt_, ptmin_, 200);
  h_ptMuonOverlappedToTrack_ = standaloneEffDir.make<TH1D>("MuonOverlappedToTrackPt", 
							   "Global muon overlapped to track p_{t} for Z -> #mu + tk", 
							   nbinsPt_, ptmin_, 200);
  
  numberOfMatchedZMuSta_ = 0;
  numberOfMatchedZMuTrack_ = 0;
  numberOfOverlappedStandAlone_ = 0;
  numberOfOverlappedTracks_ = 0;
}

void ZMuMuEfficiency::analyze(const Event& event, const EventSetup& setup) {
  Handle<CandidateCollection> zMuTrack;  
  Handle<CandMatchMap> zMuTrackMatchMap; //Map of Z made by Mu + Track
  Handle<CandidateCollection> zMuStandAlone; 
  Handle<CandMatchMap> zMuStandAloneMatchMap; //Map of Z made by Mu + StandAlone
  Handle<CandidateCollection> muons; //Collection of Muons
  Handle<CandMatchMap> muonMatchMap; 
  Handle<IsolationCollection> muonIso; 
  Handle<CandidateCollection> tracks; //Collection of Tracks
  Handle<IsolationCollection> trackIso; 
  Handle<CandidateCollection> standAlone; //Collection of StandAlone
  Handle<IsolationCollection> standAloneIso; 
  
  event.getByLabel(zMuTrack_, zMuTrack); 
  event.getByLabel(zMuStandAlone_, zMuStandAlone); 
  event.getByLabel(muons_, muons); 
  event.getByLabel(tracks_, tracks); 
  event.getByLabel(standAlone_, standAlone); 
  
  //TRACK
  if (zMuStandAlone->size() > 0) {
    event.getByLabel(zMuStandAloneMatchMap_, zMuStandAloneMatchMap); 
    event.getByLabel(muonIso_, muonIso); 
    event.getByLabel(standAloneIso_, standAloneIso); 
    event.getByLabel(muonMatchMap_, muonMatchMap); 
    for(size_t i = 0; i < zMuStandAlone->size(); ++i) { //loop on candidates
      const Candidate & zMuStaCand = (*zMuStandAlone)[i]; //the candidate
      CandidateRef zMuStaCandRef(zMuStandAlone,i);
      bool isMatched = false;
      CandMatchMap::const_iterator zMuStaMapIt = zMuStandAloneMatchMap->find(zMuStaCandRef);
      if(zMuStaMapIt != zMuStandAloneMatchMap->end()) isMatched = true;
      CandidateRef dau0 = zMuStaCand.daughter(0)->masterClone().castTo<CandidateRef>();
      CandidateRef dau1 = zMuStaCand.daughter(1)->masterClone().castTo<CandidateRef>();
      
      // Cuts
      if((dau0->pt() > ptmin_) && (dau1->pt() > ptmin_) && 
	 (fabs(dau0->eta()) < etamax_) && (fabs(dau1->eta()) < etamax_) && 
	 (zMuStaCand.mass() > zMassMin_) && (zMuStaCand.mass() < zMassMax_) && 
	 (isMatched)) {
	CandidateRef standAloneCandRef(standAlone,0);
	if(dau0.id() == standAloneCandRef.id()) {
	  standAloneMuonCandRef_ = dau0;
	  globalMuonCandRef_ = dau1;
	}
	if(dau1.id()== standAloneCandRef.id()) {
	  standAloneMuonCandRef_ = dau1;
	  globalMuonCandRef_ = dau0;
	}
	//The Z daughters are already matched!
	const double globalMuonIsolation = (*muonIso)[globalMuonCandRef_];
	const double standAloneMuonIsolation = (*standAloneIso)[standAloneMuonCandRef_];
	
	if((globalMuonIsolation < isomax_) && (standAloneMuonIsolation < isomax_)) {
	  numberOfMatchedZMuSta_++;
	  h_etaStandAlone_->Fill(standAloneMuonCandRef_->eta()); //Denominator eta for measuring track efficiency
	  h_ptStandAlone_->Fill(standAloneMuonCandRef_->pt());   //Denominator pt for measuring track eff
	  
	  for(size_t j = 0; j < muons->size() ; ++j) {
	    const Candidate & muCand = (*muons)[j]; 
	    CandidateRef muCandRef(muons, j); 
	    CandMatchMap::const_iterator muonMapIt = muonMatchMap->find(muCandRef); 
	    if((muonMapIt != muonMatchMap->end()) && (overlap_(*standAloneMuonCandRef_, muCand))) { 
	      h_etaMuonOverlappedToStandAlone_->Fill(standAloneMuonCandRef_->eta()); //Numerator eta
	      h_ptMuonOverlappedToStandAlone_->Fill(standAloneMuonCandRef_->pt());   //Numerator pt
	      numberOfOverlappedTracks_++;
	    }
	  }
	}
      }
    }
  } //end loop on Candidate
  
  //STANDALONE
  if (zMuTrack->size() > 0) {
    event.getByLabel(zMuTrackMatchMap_, zMuTrackMatchMap); 
    event.getByLabel(muonIso_, muonIso); 
    event.getByLabel(trackIso_, trackIso); 
    event.getByLabel(muonMatchMap_, muonMatchMap); 
    for(size_t i = 0; i < zMuTrack->size(); ++i) { //loop on candidates
      const Candidate & zMuTrkCand = (*zMuTrack)[i]; //the candidate
      CandidateRef zMuTrkCandRef(zMuTrack,i);
      bool isMatched = false;
      CandMatchMap::const_iterator zMuTrkMapIt = zMuTrackMatchMap->find(zMuTrkCandRef);
      if(zMuTrkMapIt != zMuTrackMatchMap->end()) isMatched = true;
      CandidateRef dau0 = zMuTrkCand.daughter(0)->masterClone().castTo<CandidateRef>();
      CandidateRef dau1 = zMuTrkCand.daughter(1)->masterClone().castTo<CandidateRef>();

      // Cuts
      if ((dau0->pt() > ptmin_) && (dau1->pt() > ptmin_) && 
	  (fabs(dau0->eta()) < etamax_) && (fabs(dau1->eta())< etamax_) && 
	  (zMuTrkCand.mass() > zMassMin_) && (zMuTrkCand.mass() < zMassMax_) && 
	  (isMatched)) {
	CandidateRef trackCandRef(tracks,0);
	if(dau0.id() == trackCandRef.id()) {
	  trackMuonCandRef_ = dau0;
	  globalMuonCandRef_ = dau1;
	}
	if(dau1.id() == trackCandRef.id()) {
	  trackMuonCandRef_ = dau1;
	  globalMuonCandRef_ = dau0;
	}
	//The Z daughters are already matched!
	const double globalMuonIsolation = (*muonIso)[globalMuonCandRef_];
	const double trackMuonIsolation = (*trackIso)[trackMuonCandRef_];
	
	if((globalMuonIsolation < isomax_) && (trackMuonIsolation < isomax_)) {
	  numberOfMatchedZMuTrack_++;
	  h_etaTrack_->Fill(trackMuonCandRef_->eta()); //Denominator eta Sta
	  h_ptTrack_->Fill(trackMuonCandRef_->pt());   //Denominator pt Sta
	  
	  for(size_t j = 0; j < muons->size() ; ++j) {
	    const Candidate & muCand = (*muons)[j];
	    CandidateRef muCandRef(muons, j); 
	    CandMatchMap::const_iterator muonMapIt = muonMatchMap->find(muCandRef); 
	    if((muonMapIt != muonMatchMap->end()) && (overlap_(*trackMuonCandRef_, muCand))) { 
	      h_etaMuonOverlappedToTrack_->Fill(trackMuonCandRef_->eta()); //Numerator sta eta
	      h_ptMuonOverlappedToTrack_->Fill(trackMuonCandRef_->pt());   //Numerator sta pt
	      numberOfOverlappedStandAlone_++;
	    }
	  }
	}
      }
    }
  } //end loop on Candidate  
} 


void ZMuMuEfficiency::endJob() {
  double efficiencySTA =(double)numberOfOverlappedStandAlone_/(double)numberOfMatchedZMuTrack_;
  double errorEff_STA = sqrt( efficiencySTA*(1 - efficiencySTA)/(double)numberOfMatchedZMuTrack_);

  double efficiencyTRACK =(double)numberOfOverlappedTracks_/(double)numberOfMatchedZMuSta_;
  double errorEff_TRACK = sqrt( efficiencyTRACK*(1 - efficiencyTRACK)/(double)numberOfMatchedZMuSta_);

  cout << "------------------------------------   Efficiency   ----------------------------- " << endl;
  cout << "numberOfOverlappedStandAlone = " << numberOfOverlappedStandAlone_ << endl; 
  cout << "numberOfMatchedZMuTrack = " << numberOfMatchedZMuTrack_ << endl; 
  cout << "numberOfOverlappedTracks = " << numberOfOverlappedTracks_ << endl; 
  cout << "numberOfMatchedZMuSta = " << numberOfMatchedZMuSta_ << endl; 
  cout << "Efficiency StandAlone = " << efficiencySTA << " +/- " << errorEff_STA << endl;
  cout << "Efficiency Track      = " << efficiencyTRACK << " +/- " << errorEff_TRACK << endl;
}
  
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ZMuMuEfficiency);
  
