/* 
 * Class for the Stepping action
 */

#ifndef LaserAlignmentSimulation_LaserSteppingAction_H
#define LaserAlignmentSimulation_LaserSteppingAction_H

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
  LaserSteppingAction(edm::ParameterSet const& theConf);
  virtual ~LaserSteppingAction();

  virtual void UserSteppingAction(const G4Step* myStep);

 private: 
  int theDebugLevel;
  double theEnergyLossScalingFactor;
};
#endif
