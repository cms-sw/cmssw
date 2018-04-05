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
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

namespace edm {
  class ConsumesCollector;
}

class BCToEFilterAlgo {
 public:
  BCToEFilterAlgo(const edm::ParameterSet&, edm::ConsumesCollector && iC);
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
  edm::EDGetTokenT<reco::GenParticleCollection> genParSource_;
};
#endif
