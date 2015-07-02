#ifndef FastSimulation_EventProducer_FamosManager_H
#define FastSimulation_EventProducer_FamosManager_H

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include <string>
#include "FastSimulation/Particle/interface/ParticleTable.h"

namespace HepMC {
  class GenEvent;
}

namespace edm { 
  class ParameterSet;
  class EventSetup;
  class Run;
  class HepMCProduct;
}

class FSimEvent;
class TrajectoryManager;
class PileUpSimulator;
class MagneticField;
class CalorimetryManager;
class RandomEngineAndDistribution;
class TrackerTopology;

// using trailing _ for private data members, m_p prefix for PSet variables (MSt)

class FamosManager
{
 public:


  /// Constructor
  FamosManager(edm::ParameterSet const & p);

  /// Destructor
  ~FamosManager();

  /// Get information from the Event Setup
  void setupGeometryAndField(edm::Run const& run, const edm::EventSetup & es);

  /// The simulated event 
  FSimEvent* simEvent() const { return mySimEvent; }

  /// The real thing is done here
  void reconstruct(const HepMC::GenEvent* evt, 
		   const TrackerTopology *tTopo,
                   RandomEngineAndDistribution const*);
  
  /// The tracker 
  TrajectoryManager * trackerManager() const {return myTrajectoryManager;}

  /// The calorimeter 
  CalorimetryManager * calorimetryManager() const {return myCalorimetry;}
  
  
 private:   

  int iEvent;
  //  const HepMC::GenEvent* myGenEvent;
  FSimEvent* mySimEvent;
  TrajectoryManager* myTrajectoryManager;
  CalorimetryManager * myCalorimetry;

 private:

  bool m_pUseMagneticField;
  bool m_Tracking;
  bool m_Calorimetry;
  bool m_Alignment;
  double weight_;    
  int m_pRunNumber;
  int m_pVerbose;
};
#endif
