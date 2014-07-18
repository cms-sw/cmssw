#ifndef PhysicsTools_JetMCAlgos_TauGenJetProducer_
#define PhysicsTools_JetMCAlgos_TauGenJetProducer_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"


/**\class TauGenJetProducer
\brief builds a GenJet from the visible daughters of each status 2 tau in the event.

\author Colin Bernet
\date   february 2008
*/
class TauGenJetProducer : public edm::EDProducer {
 public:

  explicit TauGenJetProducer(const edm::ParameterSet&);

  ~TauGenJetProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:

  /// Input PFCandidates
  edm::InputTag   inputTagGenParticles_;
  edm::EDGetTokenT<reco::GenParticleCollection>   tokenGenParticles_;

  /// if yes, neutrinos will be included, for debug purposes
  bool   includeNeutrinos_;

  /// verbose ?
  bool   verbose_;

};

#endif
