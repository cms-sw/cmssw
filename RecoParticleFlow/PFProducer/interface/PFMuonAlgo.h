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

  static bool isMuon( const reco::MuonRef& muonRef );

  static bool isLooseMuon( const reco::MuonRef& muonRef );

 private:

};

#endif
