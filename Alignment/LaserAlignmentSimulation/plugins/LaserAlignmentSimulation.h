/*
 * SimWatcher for the simulation of the Laser Alignment System of the CMS Tracker
 */

#ifndef LaserAlignmentSimulation_LaserAlignmentSimulation_H
#define LaserAlignmentSimulation_LaserAlignmentSimulation_H

#include "SimG4Core/Watcher/interface/SimWatcher.h"

#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/EndOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"
#include "SimG4Core/Notification/interface/EndOfTrack.h"

// own classes
#include "Alignment/LaserAlignmentSimulation/interface/MaterialProperties.h"
#include "Alignment/LaserAlignmentSimulation/interface/LaserPrimaryGeneratorAction.h"
#include "Alignment/LaserAlignmentSimulation/interface/LaserSteppingAction.h"

#include <map>
#include <iostream>

// Geant4 includes
#include "G4HCofThisEvent.hh"
#include "G4PrimaryParticle.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Timer.hh"
#include "G4VProcess.hh"

#include "G4ProductionCutsTable.hh"

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
  explicit LaserAlignmentSimulation(edm::ParameterSet const & theConf);
  virtual ~LaserAlignmentSimulation();
  
/*  private: */
  void update(const BeginOfRun * myRun);
  void update(const BeginOfEvent * myEvent);
  void update(const G4Step * myStep);
  void update(const EndOfEvent * myEvent);
  void update(const EndOfRun * myRun);
  void update(const BeginOfTrack * myTrack);
  void update(const EndOfTrack * myTrack);
  
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

/*   TrackerG4SimHitNumberingScheme theNumberingScheme; */
};

#endif
