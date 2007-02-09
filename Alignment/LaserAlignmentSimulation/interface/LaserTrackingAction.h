/* 
 * the Laser Tracking Action
 */

#ifndef LaserAlignmentSimulation_LaserTrackingAction_H
#define LaserAlignmentSimulation_LaserTrackingAction_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4DynamicParticle.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTypes.hh"
#include "G4Track.hh"
#include "G4TrackingManager.hh"
#include "G4UserTrackingAction.hh"

class LaserTrackingAction : public G4UserTrackingAction
{
 public:
  LaserTrackingAction(edm::ParameterSet const& theConf);
  virtual ~LaserTrackingAction();
  G4int verboseLevel;

  virtual void PreUserTrackingAction(const G4Track * theTrack);
  virtual void PostUserTrackingAction(const G4Track * theTrack);

 protected:
};
#endif
