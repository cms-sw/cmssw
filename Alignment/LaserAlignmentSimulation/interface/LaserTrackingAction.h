#ifndef LaserAlignmentSimulation_LaserTrackingAction_H
#define LaserAlignmentSimulation_LaserTrackingAction_H

/** \class LaserTrackingAction
 *  the Laser Tracking Action
 *
 *  $Date: 2007/06/11 14:44:28 $
 *  $Revision: 1.3 $
 *  \author Maarten Thomas
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4TrackingManager.hh"

class LaserTrackingAction : public G4UserTrackingAction {
public:
  /// constructor
  LaserTrackingAction(edm::ParameterSet const &theConf);
  /// destructor
  ~LaserTrackingAction() override;

  /// pre tracking action
  void PreUserTrackingAction(const G4Track *theTrack) override;
  /// post tracking action
  void PostUserTrackingAction(const G4Track *theTrack) override;

protected:
};
#endif
