#ifndef LaserAlignmentSimulation_LaserSteppingVerbose_h
#define LaserAlignmentSimulation_LaserSteppingVerbose_h

/** \class LaserSteppingVerbose
 *  Class to manage verbose stepping
 *
 *  $Date: 2007/03/20 12:00:59 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */
#include "G4SteppingVerbose.hh"

class LaserSteppingVerbose : public G4SteppingVerbose {
public:
  /// constructor
  LaserSteppingVerbose();
  /// destructor
  ~LaserSteppingVerbose() override;
  /// step information
  void StepInfo() override;
  /// tracking information
  void TrackingStarted() override;
};
#endif
