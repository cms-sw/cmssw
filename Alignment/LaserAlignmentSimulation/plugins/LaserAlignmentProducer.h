#ifndef LaserAlignmentSimulation_LaserAlignmentProducer_H
#define LaserAlignmentSimulation_LaserAlignmentProducer_H

/** \class LaserAlignmentProducer
 *  Producer to be used for the Simulation of the Laser Alignment System
 *  an empty MCHepEvent will be generated (needed by OscarProducer). The actual simulation of 
 *  the laser beams is done in the SimWatcher attached to OscarProducer
 *
 *  $Date: 2007/12/04 23:53:06 $
 *  $Revision: 1.3 $
 *  \author Maarten Thomas
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "HepMC/GenEvent.h"

//
// class decleration
//
class LaserAlignmentProducer : public edm::one::EDProducer<> {
 public:
	/// constructor
  explicit LaserAlignmentProducer(const edm::ParameterSet&);
	/// destructor
  ~LaserAlignmentProducer() override;

  
 private:
	/// produce the HepMCProduct
  void produce(edm::Event&, const edm::EventSetup&) override;
  
  // the event format itself
  HepMC::GenEvent* theEvent;
};
#endif
