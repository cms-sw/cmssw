#ifndef MuonIdentification_MuonCosmicCompatibilityFiller_h
#define MuonIdentification_MuonCosmicCompatibilityFiller_h

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
//#include "RecoMuon/MuonIdentification/interface/GlobalMuonRefitter.h"
//#include "RecoMuon/MuonIdentification/interface/GlobalMuonTrackMatcher.h"

//#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
//#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

class GlobalMuonRefitter;

class MuonCosmicCompatibilityFiller {
 public:
  MuonCosmicCompatibilityFiller(const edm::ParameterSet&);
  ~MuonCosmicCompatibilityFiller();
  
  reco::MuonCosmicCompatibility fillCompatibility( const reco::Muon& muon,edm::Event&, const edm::EventSetup&);
  
 private:
 
  float cosmicTime(const reco::Muon&) const;
  //! count matching opp side tracks: cosmics is expected to have > 0 (1 or 2 depending on config)
  unsigned int backToBack2LegCosmic(const edm::Event&, const reco::Muon&) const;
  bool isOverlappingMuon(const edm::Event&, const reco::Muon&) const;

  //! count compatible primary vertices: cosmics is expected to have 0
  unsigned int pvMatches(const edm::Event&, const reco::Muon&, bool isLoose) const;
  float isGoodCosmic(const edm::Event&, const reco::Muon&, bool CheckMuonID ) const;


  bool isMuonID( const reco::Muon& ) const;

  edm::InputTag inputMuonCollection_;
  std::vector<edm::InputTag> inputTrackCollections_;
  edm::InputTag inputCosmicMuonCollection_;
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
