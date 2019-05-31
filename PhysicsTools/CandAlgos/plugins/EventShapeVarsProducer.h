#ifndef PhysicsTools_CandAlgos_EventShapeVarsProducer_h
#define PhysicsTools_CandAlgos_EventShapeVarsProducer_h

/** \class EventShapeVarsProducer
 *
 * Produce set of event shape variables.
 * The values of different event shape variables are stored as doubles in the event.
 * They can be retrieved with InputTags like "moduleName::instanceName", where moduleName corresponds to
 * "eventShapeVarsProducer" per default and instance name specifies an individual event shape variable
 * which you wish to retrieve from the event:
 *
 *  - thrust
 *  - oblateness
 *  - isotropy
 *  - circularity
 *  - sphericity
 *  - aplanarity
 *  - C
 *  - D
 *  - Fox-Wolfram moments
 *
 *  See https://arxiv.org/pdf/hep-ph/0603175v2.pdf#page=524
 *  for an explanation of sphericity, aplanarity, the quantities C and D, thrust, oblateness, Fox-Wolfram moments.
 *
 * \author Christian Veelken, UC Davis
 *
 *
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"

class EventShapeVarsProducer : public edm::EDProducer {
public:
  explicit EventShapeVarsProducer(const edm::ParameterSet&);
  ~EventShapeVarsProducer() override {}

private:
  edm::EDGetTokenT<edm::View<reco::Candidate> > srcToken_;
  double r_;
  unsigned fwmax_;

  void beginJob() override {}
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}
};

#endif
