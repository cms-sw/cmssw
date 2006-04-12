#ifndef FastSimulation_EventProducer_FamosManager_H
#define FastSimulation_EventProducer_FamosManager_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <fstream>
#include <string>

namespace HepMC {
  class GenEvent;
}

class FSimEvent;
class TrajectoryManager;

namespace CLHEP {
  class HepLorentzVector;
}
// using trailing _ for private data members, m_p prefix for PSet variables (MSt)

class FamosManager
{
 public:


  /// Constructor
  FamosManager(edm::ParameterSet const & p);

  /// Destructor
  ~FamosManager();

  /// Get information from the Event Setup
  void setupGeometryAndField(const edm::EventSetup & es);    

  /// The generated event
  const HepMC::GenEvent* genEvent() const { return myGenEvent; };

  /// The simulated event 
  FSimEvent* simEvent() const { return mySimEvent; }

  /// The real thing is done here
  void reconstruct(const HepMC::GenEvent* evt);
  
 private:   

  int iEvent;
  const HepMC::GenEvent* myGenEvent;
  FSimEvent* mySimEvent;
  TrajectoryManager* myTrajectoryManager;
  bool m_pUseMagneticField;
  CLHEP::HepLorentzVector * vtx_;
  double weight_;    
  int m_pRunNumber;
  int m_pVerbose;

};
                       
#endif
