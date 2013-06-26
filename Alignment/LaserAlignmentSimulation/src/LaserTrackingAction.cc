/** \file LaserTrackingAction.cc
 *  
 *
 *  $Date: 2007/03/20 12:01:01 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignmentSimulation/interface/LaserTrackingAction.h"

LaserTrackingAction::LaserTrackingAction(edm::ParameterSet const& theConf) 
{
}

LaserTrackingAction::~LaserTrackingAction()
{
}

void LaserTrackingAction::PreUserTrackingAction(const G4Track * theTrack)
{
  /* *********************************************************************** */
  /* This code is called every time a new Track is created                   */
  /* *********************************************************************** */

   if ( theTrack->GetParentID()==0 )
     { fpTrackingManager->SetStoreTrajectory(true); }
   else
     { fpTrackingManager->SetStoreTrajectory(true); }

}

void LaserTrackingAction::PostUserTrackingAction(const G4Track * theTrack)
{
  /* *********************************************************************** */
  /* This code is called every time a new Track is destroyed                 */
  /* *********************************************************************** */
}
