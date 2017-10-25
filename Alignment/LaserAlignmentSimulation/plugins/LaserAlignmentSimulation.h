#ifndef LaserAlignmentSimulation_LaserAlignmentSimulation_H
#define LaserAlignmentSimulation_LaserAlignmentSimulation_H

/** \class LaserAlignmentSimulation
 *  SimWatcher for the simulation of the Laser Alignment System of the CMS Tracker
 *
 *  $Date: 2007/03/20 12:01:00 $
 *  $Revision: 1.3 $
 *  \author Maarten Thomas
 */

#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "SimG4Core/Notification/interface/Observer.h"

// own classes
#include "Alignment/LaserAlignmentSimulation/interface/MaterialProperties.h"
#include "Alignment/LaserAlignmentSimulation/interface/LaserPrimaryGeneratorAction.h"
#include "Alignment/LaserAlignmentSimulation/interface/LaserSteppingAction.h"

#include <map>
#include <iostream>

// Geant4 includes


class BeginOfRun;
class EndOfRun;
class BeginOfEvent;
class EndOfEvent;
class BeginOfTrack;
class EndOfTrack;
class G4Step;

class G4Timer;

class EventAction;
class RunAction;
class SteppingAction;
class TrackingAction;

class LaserAlignmentSimulation : public Observer<const BeginOfRun *>,
  public Observer<const BeginOfEvent *>,
  public Observer<const G4Step *>,
  public Observer<const EndOfEvent *>,
  public Observer<const EndOfRun *>,
  public Observer<const BeginOfTrack *>,
  public Observer<const EndOfTrack *>,
  public SimWatcher
{
 public:
	/// constructor
  explicit LaserAlignmentSimulation(edm::ParameterSet const & theConf);
	/// destructor
  ~LaserAlignmentSimulation() override;
  
/*  private: */
	/// observer for BeginOfRun
  void update(const BeginOfRun * myRun) override;
	/// observer for BeginOfEvent
  void update(const BeginOfEvent * myEvent) override;
	/// observer for G4Step
  void update(const G4Step * myStep) override;
	/// observer for EndOfEvent
  void update(const EndOfEvent * myEvent) override;
	/// observer for EndOfRun
  void update(const EndOfRun * myRun) override;
	/// observer for BeginOfTrack
  void update(const BeginOfTrack * myTrack) override;
	/// observer for EndOfTrack
  void update(const EndOfTrack * myTrack) override;
  
 private:
  int theDebugLevel;
  double theEnergyLossScalingFactor;
  int theMPDebug;
  double theSiAbsLengthScale;

 private:
  G4Timer * theTimer;
  MaterialProperties * theMaterialProperties;
  LaserPrimaryGeneratorAction * thePrimaryGenerator;
  LaserSteppingAction * theSteppingAction;

  int theBarrelHits;
  int theEndcapHits;

  edm::ParameterSet theParameterSet;
/*   TrackerG4SimHitNumberingScheme theNumberingScheme; */
};

#endif
