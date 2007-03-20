#ifndef LaserAlignmentSimulation_LaserSteppingVerbose_h
#define LaserAlignmentSimulation_LaserSteppingVerbose_h

/** \class LaserSteppingVerbose
 *  Class to manage verbose stepping
 *
 *  $Date: Mon Mar 19 12:12:50 CET 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */
#include "G4SteppingVerbose.hh"
#include "G4SteppingManager.hh"
#include "G4UnitsTable.hh"

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
