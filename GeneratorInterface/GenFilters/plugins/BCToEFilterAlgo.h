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

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

class BCToEFilterAlgo {
public:
  BCToEFilterAlgo(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
  ~BCToEFilterAlgo();

  bool filter(const edm::Event& iEvent) const;

  bool hasBCAncestors(const reco::GenParticle& gp) const;

private:
  bool isBCHadron(const reco::GenParticle& gp) const;
  bool isBCMeson(const reco::GenParticle& gp) const;
  bool isBCBaryon(const reco::GenParticle& gp) const;

  //filter parameters:
  const float maxAbsEta_;
  const float eTThreshold_;
  const edm::EDGetTokenT<reco::GenParticleCollection> genParSource_;
};
#endif
