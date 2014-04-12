#ifndef BCToEFilterAlgo_h
#define BCToEFilterAlgo_h

/** \class BCToEFilterAlgo
 *
 *  BCToEFilterAlgo
 *  returns true for events that have an electron, above configurable eT threshold and within |eta|<2.5, that has an ancestor of a b or c quark
 *
 * \author J Lamb, UCSB
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



class BCToEFilterAlgo {
 public:
  BCToEFilterAlgo(const edm::ParameterSet&);
  ~BCToEFilterAlgo();
  
  bool filter(const edm::Event& iEvent);

  bool hasBCAncestors(const reco::GenParticle& gp);

 private:

  bool isBCHadron(const reco::GenParticle& gp);
  bool isBCMeson(const reco::GenParticle& gp);
  bool isBCBaryon(const reco::GenParticle& gp);

  

 private:
  //constants:
  float FILTER_ETA_MAX_;
  //filter parameters:
  float eTThreshold_;
  edm::InputTag genParSource_;
  
};
#endif
