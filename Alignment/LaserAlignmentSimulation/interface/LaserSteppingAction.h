#ifndef LaserAlignmentSimulation_LaserSteppingAction_H
#define LaserAlignmentSimulation_LaserSteppingAction_H

/** \class LaserSteppingAction
 *  Class for the Stepping action
 *
 *  $Date: 2007/03/20 12:00:59 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4SteppingManager.hh"

class LaserSteppingAction : public G4UserSteppingAction {
public:
  /// constructor
  LaserSteppingAction(edm::ParameterSet const &theConf);
  /// destructor
  ~LaserSteppingAction() override;
  /// stepping action: set energydeposit when a photon is absorbed in a Si
  /// module
  void UserSteppingAction(const G4Step *myStep) override;

private:
  int theDebugLevel;
  double theEnergyLossScalingFactor;
};
#endif
