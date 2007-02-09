/* 
 * Source to be used for the Simulation of the Laser Alignment System
 * an empty MCHepEvent will be generated (needed by OscarProducer). The actual simulation of 
 * the laser beams is done in the SimWatcher attached to OscarProducer
 */

#ifndef LaserAlignmentSimulation_LaserAlignmentSource_H
#define LaserAlignmentSimulation_LaserAlignmentSource_H

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/HepMC/GenEvent.h"

//
// class decleration
//

class LaserAlignmentSource : public edm::GeneratedInputSource {
 public:
  explicit LaserAlignmentSource(const edm::ParameterSet&, const edm::InputSourceDescription&);
  ~LaserAlignmentSource();

  
 private:
  virtual bool produce(edm::Event&);
  
  // the event format itself
  HepMC::GenEvent* theEvent;
};
#endif
