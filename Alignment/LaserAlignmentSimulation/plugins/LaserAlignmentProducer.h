#ifndef LaserAlignmentSimulation_LaserAlignmentProducer_H
#define LaserAlignmentSimulation_LaserAlignmentProducer_H

/** \class LaserAlignmentProducer
 *  Producer to be used for the Simulation of the Laser Alignment System
 *  an empty MCHepEvent will be generated (needed by OscarProducer). The actual simulation of 
 *  the laser beams is done in the SimWatcher attached to OscarProducer
 *
 *  $Date: 2012/11/02 19:05:24 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "HepMC/GenEvent.h"

//
// class decleration
//
class LaserAlignmentProducer : public edm::EDProducer {
 public:
	/// constructor
  explicit LaserAlignmentProducer(const edm::ParameterSet&);
	/// destructor
  ~LaserAlignmentProducer();

  
 private:
	/// produce the HepMCProduct
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
  // the event format itself
  HepMC::GenEvent* theEvent;
};
#endif
