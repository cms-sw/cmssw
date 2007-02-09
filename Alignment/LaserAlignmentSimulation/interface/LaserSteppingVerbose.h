/*
 * Class to manage verbose stepping
 */

#ifndef LaserAlignmentSimulation_LaserSteppingVerbose_h
#define LaserAlignmentSimulation_LaserSteppingVerbose_h

#include "G4SteppingVerbose.hh"
#include "G4SteppingManager.hh"
#include "G4UnitsTable.hh"

class LaserSteppingVerbose : public G4SteppingVerbose
{
public:
  LaserSteppingVerbose();    // constructor
  ~LaserSteppingVerbose();   // destructor

  void StepInfo();
  void TrackingStarted();
};
#endif
