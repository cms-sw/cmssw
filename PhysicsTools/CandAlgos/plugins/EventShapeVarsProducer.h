#ifndef PhysicsTools_CandAlgos_EventShapeVarsProducer_h  
#define PhysicsTools_CandAlgos_EventShapeVarsProducer_h

/** \class EventShapeVarsProducer
 *
 * Produce set of event shape variables.
 * The values of different event shape variables are stored as doubles in the event.
 * They can be retrieved via:
 *
 *   edm::Handle<double> specificEventShapeVar;
 *   evt.getByLabel("moduleName::instanceName", specificEventShapeVar);
 *
 * where moduleName corresponds to "eventShapeVarsProducer" per default
 * and instance name specifies an individual event shape variable which you wish to retrieve from the event:
 *
 *  - thrust
 *  - oblateness
 *  - isotropy
 *  - circularity
 *  - sphericity
 *  - aplanarity
 *  - C
 *  - D
 *
 *  See http://cepa.fnal.gov/psm/simulation/mcgen/lund/pythia_manual/pythia6.3/pythia6301/node213.html
 *  ( http://cepa.fnal.gov/psm/simulation/mcgen/lund/pythia_manual/pythia6.3/pythia6301/node214.html )
 *  for an explanation of sphericity, aplanarity and the quantities C and D (thrust and oblateness).
 *
 * \author Christian Veelken, UC Davis
 *
 * \version $Revision: 1.3 $
 *
 * $Id: EventShapeVarsProducer.h,v 1.3 2010/02/11 00:12:54 wmtan Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EventShapeVarsProducer : public edm::EDProducer
{
 public:
  
  explicit EventShapeVarsProducer(const edm::ParameterSet&);
  ~EventShapeVarsProducer() {}
  
 private:

  edm::InputTag src_;
  double r_;

  void beginJob() {}
  void produce(edm::Event&, const edm::EventSetup&);
  void endJob() {}

};

#endif  


