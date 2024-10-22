#ifndef PhysicsTools_JetMCAlgos_TauGenJetProducer_
#define PhysicsTools_JetMCAlgos_TauGenJetProducer_

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

/**\class TauGenJetProducer
\brief builds a GenJet from the visible daughters of each status 2 tau in the event.

\author Colin Bernet
\date   february 2008
*/
class TauGenJetProducer : public edm::global::EDProducer<> {
public:
  explicit TauGenJetProducer(const edm::ParameterSet&);

  ~TauGenJetProducer() override;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  /// Input PFCandidates
  const edm::EDGetTokenT<reco::GenParticleCollection> tokenGenParticles_;

  /// if yes, neutrinos will be included, for debug purposes
  const bool includeNeutrinos_;

  /// verbose ?
  const bool verbose_;
};

#endif
