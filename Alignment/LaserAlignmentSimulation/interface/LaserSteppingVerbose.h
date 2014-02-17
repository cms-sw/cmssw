#ifndef LaserAlignmentSimulation_LaserSteppingVerbose_h
#define LaserAlignmentSimulation_LaserSteppingVerbose_h

/** \class LaserSteppingVerbose
 *  Class to manage verbose stepping
 *
 *  $Date: 2007/06/11 14:44:28 $
 *  $Revision: 1.3 $
 *  \author Maarten Thomas
 */
#include "G4SteppingVerbose.hh"

class LaserSteppingVerbose : public G4SteppingVerbose
{
public:
	/// constructor
  LaserSteppingVerbose();
  /// destructor
  ~LaserSteppingVerbose();
	/// step information 
  void StepInfo();
	/// tracking information
  void TrackingStarted();
};
#endif
