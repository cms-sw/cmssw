/** \file LaserTrackingAction.cc
 *
 *
 *  $Date: Mon Mar 19 12:21:52 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignmentSimulation/interface/LaserTrackingAction.h"

LaserTrackingAction::LaserTrackingAction(edm::ParameterSet const &theConf) {}

LaserTrackingAction::~LaserTrackingAction() {}

void LaserTrackingAction::PreUserTrackingAction(const G4Track *theTrack) {
  /* *********************************************************************** */
  /* This code is called every time a new Track is created                   */
  /* *********************************************************************** */

  if (theTrack->GetParentID() == 0) {
    fpTrackingManager->SetStoreTrajectory(true);
  } else {
    fpTrackingManager->SetStoreTrajectory(true);
  }
}

void LaserTrackingAction::PostUserTrackingAction(const G4Track *theTrack) {
  /* *********************************************************************** */
  /* This code is called every time a new Track is destroyed                 */
  /* *********************************************************************** */
}
