#ifndef LaserAlignmentSimulation_LaserTrackingAction_H
#define LaserAlignmentSimulation_LaserTrackingAction_H

/** \class LaserTrackingAction
 *  the Laser Tracking Action
 *
 *  $Date: Mon Mar 19 12:17:36 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

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
	/// constructor
  LaserTrackingAction(edm::ParameterSet const& theConf);
	/// destructor
  virtual ~LaserTrackingAction();
  G4int verboseLevel;

	/// pre tracking action
  virtual void PreUserTrackingAction(const G4Track * theTrack);
	/// post tracking action
  virtual void PostUserTrackingAction(const G4Track * theTrack);

 protected:
};
#endif
