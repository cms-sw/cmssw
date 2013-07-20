#ifndef LaserAlignmentSimulation_LaserAlignmentSource_H
#define LaserAlignmentSimulation_LaserAlignmentSource_H

/** \class LaserAlignmentSource
 *  Source to be used for the Simulation of the Laser Alignment System
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
#include "FWCore/Framework/interface/GeneratedInputSource.h"

#include "HepMC/GenEvent.h"

//
// class decleration
//
class LaserAlignmentSource : public edm::GeneratedInputSource {
 public:
	/// constructor
  explicit LaserAlignmentSource(const edm::ParameterSet&, const edm::InputSourceDescription&);
	/// destructor
  ~LaserAlignmentSource();

  
 private:
	/// produce the HepMCProduct
  virtual bool produce(edm::Event&);
  
  // the event format itself
  HepMC::GenEvent* theEvent;
};
#endif
