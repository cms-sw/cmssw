#ifndef HeavyQuarkFromMPIFilterAlgo_h
#define HeavyQuarkFromMPIFilterAlgo_h

/** \class HeavyQuarkFromMPIFilterAlgo
 *
 ************************************************************/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"



class HeavyQuarkFromMPIFilterAlgo {
 public:
  HeavyQuarkFromMPIFilterAlgo(const edm::ParameterSet&);
  ~HeavyQuarkFromMPIFilterAlgo();
  
  bool filter(const edm::Event& iEvent);
  bool hasMPIAncestor( const reco::GenParticle*); 

 private:
  int HeavyQuarkFlavour;
 edm::InputTag genParSource_; 
};
#endif
