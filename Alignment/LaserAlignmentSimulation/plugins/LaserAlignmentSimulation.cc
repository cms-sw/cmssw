/** \file LaserAlignmentSimulation.cc
 *  SimWatcher for the simulation of the Laser Alignment System of the CMS Tracker
 *
 *  $Date: 2010/02/25 00:27:58 $
 *  $Revision: 1.6 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignmentSimulation/plugins/LaserAlignmentSimulation.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h" 
#include "SimG4Core/Notification/interface/EndOfRun.h" 
#include "SimG4Core/Notification/interface/BeginOfEvent.h" 
#include "SimG4Core/Notification/interface/EndOfEvent.h" 
#include "G4SDManager.hh" 
#include "G4Step.hh" 
#include "G4Timer.hh" 
#include "G4VProcess.hh" 

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "SimG4CMS/Tracker/interface/TkAccumulatingSensitiveDetector.h"
#include "G4StepPoint.hh" 



LaserAlignmentSimulation::LaserAlignmentSimulation(edm::ParameterSet const& theConf) 
  : theDebugLevel(theConf.getUntrackedParameter<int>("DebugLevel",0)),
    theEnergyLossScalingFactor(theConf.getUntrackedParameter<double>("EnergyLossScalingFactor",1.0)),
    theMPDebug(theConf.getUntrackedParameter<int>("MaterialPropertiesDebugLevel",0)),
    theSiAbsLengthScale(theConf.getUntrackedParameter<double>("SiAbsorptionLengthScalingFactor",1.0)),
    theTimer(), 
    theMaterialProperties(),
    thePrimaryGenerator(), theSteppingAction(),
    theBarrelHits(0), theEndcapHits(0),
		theParameterSet(theConf)
{

  // make some noise
  edm::LogInfo("SimLaserAlignmentSimulation") << " *****     AC1CMS: Configuration from ParameterSet      ***** " 
				    << "\n  AC1CMS: theDebugLevel               = " << theDebugLevel 
				    << "\n  AC1CMS: theEnergyLossScalingFactor  = " << theEnergyLossScalingFactor 
				    << "\n  AC1CMS: theMPDebugLevel             = " << theMPDebug
				    << "\n  AC1CMS: theSiAbsLengthScalingFactor = " << theSiAbsLengthScale;

  // declare timer
  theTimer = new G4Timer;
}

LaserAlignmentSimulation::~LaserAlignmentSimulation() 
{
  if ( theMaterialProperties != 0 )        { delete theMaterialProperties; }
  if ( theSteppingAction != 0 )            { delete theSteppingAction; }
  if ( thePrimaryGenerator != 0 )          { delete thePrimaryGenerator; }
  if ( theTimer != 0 )                     { delete theTimer; }
}

void LaserAlignmentSimulation::update(const BeginOfRun * myRun)
{
  LogDebug("SimLaserAlignmentSimulation") << "<LaserAlignmentSimulation::update(const BeginOfRun * myRun)>"
				<< "\n *****     AC1CMS: Start of Run: " << (*myRun)()->GetRunID() << "     ***** ";

  // start timer
  theTimer->Start();


  // the PrimaryGeneratorAction: defines the used particlegun for the Laser events
  thePrimaryGenerator = new LaserPrimaryGeneratorAction(theParameterSet);

  // the UserSteppingAction: at the moment this prints only some information
  theSteppingAction = new LaserSteppingAction(theParameterSet);

  // construct your own material properties for setting refractionindex and so on
  theMaterialProperties = new MaterialProperties(theMPDebug, theSiAbsLengthScale);

  // list the tree of sensitive detectors
  if (theDebugLevel >= 1)
    {
      G4SDManager * theSDManager = G4SDManager::GetSDMpointer();
      theSDManager->ListTree();
    }
}

void LaserAlignmentSimulation::update(const BeginOfEvent * myEvent) 
{
  LogDebug("SimLaserAlignmentSimulation") << "<LaserAlignmentSimulation::update(const BeginOfEvent * myEvent)>"
				<< "\n AC1CMS: Event number = " << (*myEvent)()->GetEventID();

  // some statistics for this event
  theBarrelHits = 0;
  theEndcapHits = 0;

  // generate the Primaries
  thePrimaryGenerator->GeneratePrimaries((G4Event*)(*myEvent)());
}

void LaserAlignmentSimulation::update(const BeginOfTrack * myTrack)
{
}

void LaserAlignmentSimulation::update(const G4Step * myStep) 
{
  LogDebug("SimLaserAlignmentSimulationStepping") << "<LaserAlignmentSimulation::update(const G4Step * myStep)>";

  G4Step * theStep = const_cast<G4Step*>(myStep);

  // do the LaserSteppingAction
  theSteppingAction->UserSteppingAction(theStep);

  // Trigger sensitive detector manually since photon is absorbed
  if ( ( theStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName()== "OpAbsorption" ) )
    {
      LogDebug("SimLaserAlignmentSimulationStepping") << "<LaserAlignmentSimulation::update(const G4Step*)>: Photon was absorbed! ";
      
      
      if ( theStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetSensitiveDetector() )
	{
	  LogDebug("SimLaserAlignmentSimulationStepping") << " AC1CMS: Setting the EnergyLoss to " << theStep->GetTotalEnergyDeposit() 
						<< "\n AC1CMS: The z position is " << theStep->GetPreStepPoint()->GetPosition().z()
						<< "\n AC1CMS: the Sensitive Detector: " 
						<< theStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetSensitiveDetector()->GetName()
						<< "\n AC1CMS: the Material: " << theStep->GetPreStepPoint()->GetMaterial()->GetName()
						<< "\n AC1CMS: the Logical Volume: " 
						<< theStep->GetPostStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName();

	  if (theStep->GetTotalEnergyDeposit() > 0.0)
	    {
	      // process a hit
	      TkAccumulatingSensitiveDetector * theSD = (TkAccumulatingSensitiveDetector*)
		(theStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetSensitiveDetector());
	      
	      theSD->ProcessHits(theStep, ((G4TouchableHistory *)(theStep->GetPreStepPoint()->GetTouchable())));


	      // some statistics for this event
	      if ( ( theStep->GetPostStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName() == "TECModule3RphiActive" ) || 
		   ( theStep->GetPostStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName() == "TECModule5RphiActive" ) )
		{
		  theEndcapHits++;
		}
	      else if ( ( theStep->GetPostStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName() == "TOBActiveSter0" ) ||
			( theStep->GetPostStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName() == "TOBActiveRphi0" ) ||
			( theStep->GetPostStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName() == "TIBActiveRphi2" ) )
		{
		  theBarrelHits++;
		}
	    }
	}
      else 
	{
	  LogDebug("SimLaserAlignmentSimulationStepping") << " AC1CMS: No SensitiveDetector available for this Step ... No Hit created :-( "
						<< "\n AC1CMS: The Material was: " << theStep->GetPreStepPoint()->GetMaterial()->GetName(); 
	}
    }
}

void LaserAlignmentSimulation::update(const EndOfTrack * myTrack)
{
}

void LaserAlignmentSimulation::update(const EndOfEvent * myEvent) 
{
  LogDebug("SimLaserAlignmentSimulation") << "<LaserAlignmentSimulation::update(const EndOfEvent * myEvent)>"
				<< "\n AC1CMS: End of Event " << (*myEvent)()->GetEventID();

  // some statistics for this event
  edm::LogInfo("SimLaserAlignmentSimulation") << " *** Number of Hits: " << theBarrelHits << " / " << theEndcapHits
				    << " (Barrel / Endcaps) *** ";
}

void LaserAlignmentSimulation::update(const EndOfRun * myRun)
{
  LogDebug("SimLaserAlignmentSimulation") << "<LaserAlignmentSimulation::update(const EndOfRun * myRun)>";

  // stop timer
  theTimer->Stop();
  edm::LogInfo("SimLaserAlignmentSimulation") << " AC1CMS: Number of Events = " << (*myRun)()->GetNumberOfEventToBeProcessed()
				    << " " << *theTimer 
				    << " *****     AC1CMS: End of Run: " << (*myRun)()->GetRunID() << "     ***** ";
}

// register a SimWatcher to get the Observer signals from OscarProducer
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"


DEFINE_SIMWATCHER (LaserAlignmentSimulation);
