#ifndef GlobalTrackingTools_GlobalCosmicCompatibilityFiller_h
#define GlobalTrackingTools_GlobalCosmicCompatibilityFiller_h

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonCosmicCompatibility.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
//#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonRefitter.h"
//#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonTrackMatcher.h"

//#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
//#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

class GlobalMuonRefitter;

class GlobalCosmicCompatibilityFiller {
 public:
  GlobalCosmicCompatibilityFiller(const edm::ParameterSet&);
  ~GlobalCosmicCompatibilityFiller();
  
  reco::MuonCosmicCompatibility fillCompatibility( const reco::Muon& muon,edm::Event&, const edm::EventSetup&);
  
 private:
 
  float cosmicTime(const reco::Muon&) const;
  bool backToBack2LegCosmic(const edm::Event&, const reco::Muon&) const;
  bool isOverlappingMuon(const edm::Event&, const reco::Muon&) const;
  bool isCosmicVertex(const edm::Event&, const reco::Muon&) const;
  bool isIpCosmic(const edm::Event&, const reco::Muon&, bool isLoose) const;
  float isGoodCosmic(const edm::Event&, const reco::Muon&, bool CheckMuonID ) const;
  bool isMuonID( const reco::Muon& ) const;

  template <class T> double max(const T& a, const T& b ) const{
    return (b<a)?a:b;
  }
  template <class T> double angleBetween(const T& lhs, const T& rhs) const {
    GlobalVector mom1(lhs.px(), lhs.py(), lhs.pz());
    GlobalVector mom2(rhs.px(), rhs.py(), rhs.pz());
    
    GlobalVector dmom = mom1 - mom2;
    return acos( ( mom1.mag() * mom1.mag() + mom2.mag() * mom2.mag() - dmom.mag() * dmom.mag() ) / (2*mom1.mag()*mom2.mag() ));     
  }


  edm::InputTag inputMuonCollection_;
  edm::InputTag inputTrackCollection_;
  edm::InputTag inputCosmicCollection_;
  edm::InputTag inputVertexCollection_;
  MuonServiceProxy* theService;

  double maxdxyLoose_;
  double maxdzLoose_;
  double maxdxyTight_;
  double maxdzTight_;
  double minNDOF_;
  double minvProb_;
  double deltaPhi_;
  double deltaPt_;
  double offTimePos_;
  double offTimeNeg_; 
  double ipThreshold_;
  double angleThreshold_;
  int sharedHits_;
  double sharedFrac_;

};
#endif
