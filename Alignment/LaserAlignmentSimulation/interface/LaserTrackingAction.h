#ifndef LaserAlignmentSimulation_LaserTrackingAction_H
#define LaserAlignmentSimulation_LaserTrackingAction_H

/** \class LaserTrackingAction
 *  the Laser Tracking Action
 *
 *  $Date: 2011/09/16 06:40:35 $
 *  $Revision: 1.4 $
 *  \author Maarten Thomas
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4TrackingManager.hh"

class LaserTrackingAction : public G4UserTrackingAction
{
 public:
	/// constructor
  LaserTrackingAction(edm::ParameterSet const& theConf);
	/// destructor
  virtual ~LaserTrackingAction();

	/// pre tracking action
  virtual void PreUserTrackingAction(const G4Track * theTrack);
	/// post tracking action
  virtual void PostUserTrackingAction(const G4Track * theTrack);

 protected:
};
#endif
