#ifndef LaserAlignmentSimulation_LaserSteppingAction_H
#define LaserAlignmentSimulation_LaserSteppingAction_H

/** \class LaserSteppingAction
 *  Class for the Stepping action
 *
 *  $Date: Mon Mar 19 12:11:42 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "globals.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTypes.hh"
#include "G4Step.hh"
#include "G4SteppingManager.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4TrackStatus.hh"
#include "G4UserSteppingAction.hh"
#include "G4VPhysicalVolume.hh"

class LaserSteppingAction : public G4UserSteppingAction
{
 public:
	/// constructor
  LaserSteppingAction(edm::ParameterSet const& theConf);
	/// destructor
  virtual ~LaserSteppingAction();
	/// stepping action: set energydeposit when a photon is absorbed in a Si module
  virtual void UserSteppingAction(const G4Step* myStep);

 private: 
  int theDebugLevel;
  double theEnergyLossScalingFactor;
};
#endif
