#ifndef RecoParticleFlow_PFProducer_PFMuonAlgo_h
#define RecoParticleFlow_PFProducer_PFMuonAlgo_h 

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/MuonReco/interface/Muon.h"

class PFMuonAlgo {

 public:

  /// constructor
  PFMuonAlgo() {;}

  /// destructor
  virtual ~PFMuonAlgo() {;}
  
  /// Check if a block element is a muon
  static bool isMuon( const reco::PFBlockElement& elt );

  static bool isLooseMuon( const reco::PFBlockElement& elt );

  static bool isGlobalTightMuon( const reco::PFBlockElement& elt );

  static bool isGlobalLooseMuon( const reco::PFBlockElement& elt );

  static bool isTrackerTightMuon( const reco::PFBlockElement& elt );

  static bool isTrackerLooseMuon( const reco::PFBlockElement& elt );

  static bool isIsolatedMuon( const reco::PFBlockElement& elt );

  static bool isMuon( const reco::MuonRef& muonRef );  

  static bool isLooseMuon( const reco::MuonRef& muonRef );

  static bool isGlobalTightMuon( const reco::MuonRef& muonRef );

  static bool isGlobalLooseMuon( const reco::MuonRef& muonRef );

  static bool isTrackerTightMuon( const reco::MuonRef& muonRef );
  
  static bool isTrackerLooseMuon( const reco::MuonRef& muonRef );
  
  static bool isIsolatedMuon( const reco::MuonRef& muonRef );

  static bool isTightMuonPOG(const reco::MuonRef& muonRef);

  static void printMuonProperties( const reco::MuonRef& muonRef );
  
 private:

};

#endif
