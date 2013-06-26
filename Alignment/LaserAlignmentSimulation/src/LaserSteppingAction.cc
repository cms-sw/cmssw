/** \file LaserSteppingAction.cc
 *  
 *
 *  $Date: 2009/09/30 18:33:55 $
 *  $Revision: 1.4 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignmentSimulation/interface/LaserSteppingAction.h"
#include "G4ParticleTypes.hh" 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

LaserSteppingAction::LaserSteppingAction(edm::ParameterSet const& theConf) 
  : theDebugLevel(theConf.getUntrackedParameter<int>("DebugLevel",0)),
    theEnergyLossScalingFactor(theConf.getUntrackedParameter<double>("EnergyLossScalingFactor",1.0))
{
}

LaserSteppingAction::~LaserSteppingAction()
{
}

void LaserSteppingAction::UserSteppingAction(const G4Step * myStep)
{
  G4Step * theStep = const_cast<G4Step*>(myStep);

  G4Track * theTrack = theStep->GetTrack();

  // some debug info
  {
    G4TrackStatus isGood = theTrack->GetTrackStatus();
    
    LogDebug("LaserAlignmentSimulationStepping") << "<LaserSteppingAction::UserSteppingAction(const G4Step *)>: AC1CMS: the PreStep Material = " 
					  << theStep->GetPreStepPoint()->GetMaterial()->GetName()
					  << "\n<LaserSteppingAction::UserSteppingAction(const G4Step *)>: AC1CMS: The Track Status = " << isGood;
    if ( isGood == fStopAndKill ) 
      LogDebug("LaserAlignmentSimulationStepping") << "<LaserSteppingAction::UserSteppingAction(const G4Step *)>: AC1CMS: Track Status = fStopAndKill ";
      
    if ( theStep->GetPreStepPoint()->GetProcessDefinedStep() != 0 )
      LogDebug("LaserAlignmentSimulationStepping") << "<LaserSteppingAction::UserSteppingAction(const G4Step *)>: AC1CMS: PreStep Process  = " 
					    << theStep->GetPreStepPoint()->GetProcessDefinedStep()->GetProcessName();
    if ( theStep->GetPostStepPoint()->GetProcessDefinedStep() != 0 )
      LogDebug("LaserAlignmentSimulationStepping") << "<LaserSteppingAction::UserSteppingAction(const G4Step *)>: AC1CMS: PostStep Process = " 
					    << theStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();
  }

  // ***********************************************************************************************************
  // Set the EnergyDeposit if the photon is absorbed by a active sensor
  if ( ( theStep->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName()== "OpAbsorption" ) )
    {
      LogDebug("LaserAlignmentStepping") << "<LaserSteppingAction::UserSteppingAction(const G4Step*)>: Photon was absorbed! ";
      
      if ( theStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetSensitiveDetector() )
	{
	  double EnergyLoss = theEnergyLossScalingFactor * theTrack->GetTotalEnergy();
	  
	  // use different energy deposit for the discs depending on the z-position to simulate the variable laser power
	  // Disc 1 TEC2TEC
	  if ( ( (theStep->GetPreStepPoint()->GetPosition().z() > 1262.5 && theStep->GetPreStepPoint()->GetPosition().z() < 1382.5 )
                 || (theStep->GetPreStepPoint()->GetPosition().z() < -1262.5 && theStep->GetPreStepPoint()->GetPosition().z() > -1382.5 ) )
               && ( ( ( theStep->GetPreStepPoint()->GetPosition().phi() > 1.285 && theStep->GetPreStepPoint()->GetPosition().phi() < 1.295 ) 
                      || ( theStep->GetPreStepPoint()->GetPosition().phi() > 1.84 && theStep->GetPreStepPoint()->GetPosition().phi() < 1.86 )
                      || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 3.63 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 3.66 )
                      || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 5.20 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 5.23 )
                      || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 5.76 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 5.80 ) ) ) )
 	    { theStep->AddTotalEnergyDeposit(EnergyLoss); } // Disc 1 TEC2TEC
	  // Disc 1
	  else if ( (theStep->GetPreStepPoint()->GetPosition().z() > 1262.5 && theStep->GetPreStepPoint()->GetPosition().z() < 1382.5 )
		    || (theStep->GetPreStepPoint()->GetPosition().z() < -1262.5 && theStep->GetPreStepPoint()->GetPosition().z() > -1382.5 ) )
 	    { theStep->AddTotalEnergyDeposit(EnergyLoss/(0.2*0.2*0.2*0.2)); } // Disc 1
	  // Disc 2 TEC2TEC
	  else if ( ( (theStep->GetPreStepPoint()->GetPosition().z() > 1402.5 && theStep->GetPreStepPoint()->GetPosition().z() < 1522.5 )
                      || (theStep->GetPreStepPoint()->GetPosition().z() < -1402.5 && theStep->GetPreStepPoint()->GetPosition().z() > -1522.5 ) )
		    && ( ( ( theStep->GetPreStepPoint()->GetPosition().phi() > 1.285 && theStep->GetPreStepPoint()->GetPosition().phi() < 1.295 ) 
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() > 1.84 && theStep->GetPreStepPoint()->GetPosition().phi() < 1.86 )
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 3.63 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 3.66 )
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 5.20 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 5.23 )
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 5.76 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 5.80 ) ) ) )
 	    { theStep->AddTotalEnergyDeposit(EnergyLoss/(0.2)); } // Disc 2 TEC2TEC
	  // Disc 2
	  else if ( (theStep->GetPreStepPoint()->GetPosition().z() > 1402.5 && theStep->GetPreStepPoint()->GetPosition().z() < 1522.5 )
		    || (theStep->GetPreStepPoint()->GetPosition().z() < -1402.5 && theStep->GetPreStepPoint()->GetPosition().z() > -1522.5 ) )
	    { theStep->AddTotalEnergyDeposit(EnergyLoss/(0.2*0.2*0.2)); } // Disc 2
	  // Disc 3 TEC2TEC
	  else if ( ( (theStep->GetPreStepPoint()->GetPosition().z() > 1542.5 && theStep->GetPreStepPoint()->GetPosition().z() < 1662.5 )
		    || (theStep->GetPreStepPoint()->GetPosition().z() < -1542.5 && theStep->GetPreStepPoint()->GetPosition().z() > -1662.5 ) ) 
		    && ( ( ( theStep->GetPreStepPoint()->GetPosition().phi() > 1.285 && theStep->GetPreStepPoint()->GetPosition().phi() < 1.295 ) 
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() > 1.84 && theStep->GetPreStepPoint()->GetPosition().phi() < 1.86 )
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 3.63 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 3.66 )
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 5.20 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 5.23 )
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 5.76 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 5.80 ) ) ) )
	    { theStep->AddTotalEnergyDeposit(EnergyLoss/(0.2*0.2)); } // Disc 3 TEC2TEC
	  // Disc 3
	  else if ( (theStep->GetPreStepPoint()->GetPosition().z() > 1542.5 && theStep->GetPreStepPoint()->GetPosition().z() < 1662.5 )
		    || (theStep->GetPreStepPoint()->GetPosition().z() < -1542.5 && theStep->GetPreStepPoint()->GetPosition().z() > -1662.5 ) )
	    { theStep->AddTotalEnergyDeposit(EnergyLoss/(0.2*0.2)); } // Disc 3
	  // Disc 4 TEC2TEC
	  else if ( ( (theStep->GetPreStepPoint()->GetPosition().z() > 1682.5 && theStep->GetPreStepPoint()->GetPosition().z() < 1802.5 )
                      || (theStep->GetPreStepPoint()->GetPosition().z() < -1682.5 && theStep->GetPreStepPoint()->GetPosition().z() > -1802.5 ) ) 
		    && ( ( ( theStep->GetPreStepPoint()->GetPosition().phi() > 1.285 && theStep->GetPreStepPoint()->GetPosition().phi() < 1.295 ) 
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() > 1.84 && theStep->GetPreStepPoint()->GetPosition().phi() < 1.86 )
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 3.63 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 3.66 )
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 5.20 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 5.23 )
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 5.76 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 5.80 ) ) ) )
	    { theStep->AddTotalEnergyDeposit(EnergyLoss/(0.2*0.2*0.2)); } // Disc 4 TEC2TEC
	  // Disc 4
	  else if ( (theStep->GetPreStepPoint()->GetPosition().z() > 1682.5 && theStep->GetPreStepPoint()->GetPosition().z() < 1802.5 )
		    || (theStep->GetPreStepPoint()->GetPosition().z() < -1682.5 && theStep->GetPreStepPoint()->GetPosition().z() > -1802.5 ) )
	    { theStep->AddTotalEnergyDeposit(EnergyLoss/(0.2)); } // Disc 4
	  // Disc 5 TEC2TEC
	  else if ( ( ( theStep->GetPreStepPoint()->GetPosition().z() > 1822.5 && theStep->GetPreStepPoint()->GetPosition().z() < 1942.5 )
		    || ( theStep->GetPreStepPoint()->GetPosition().z() < -1822.5 && theStep->GetPreStepPoint()->GetPosition().z() > -1942.5 ) )
		    && ( ( ( theStep->GetPreStepPoint()->GetPosition().phi() > 1.285 && theStep->GetPreStepPoint()->GetPosition().phi() < 1.295 ) 
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() > 1.84 && theStep->GetPreStepPoint()->GetPosition().phi() < 1.86 )
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 3.63 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 3.66 )
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 5.20 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 5.23 )
                           || ( theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI > 5.76 && theStep->GetPreStepPoint()->GetPosition().phi() + 2.0*M_PI < 5.80 ) ) ) )
	    { theStep->AddTotalEnergyDeposit(EnergyLoss/(0.2*0.2*0.2*0.2)); } // Disc 5 TEC2TEC
	  // Disc 5
	  else if ( (theStep->GetPreStepPoint()->GetPosition().z() > 1822.5 && theStep->GetPreStepPoint()->GetPosition().z() < 1942.5 )
		    || (theStep->GetPreStepPoint()->GetPosition().z() < -1822.5 && theStep->GetPreStepPoint()->GetPosition().z() > -1942.5 ) )
	    { theStep->AddTotalEnergyDeposit(EnergyLoss); } // Disc 5
	  // Disc 6
	  else if ( (theStep->GetPreStepPoint()->GetPosition().z() > 1997.5 && theStep->GetPreStepPoint()->GetPosition().z() < 2117.5 )
		    || (theStep->GetPreStepPoint()->GetPosition().z() < -1997.5 && theStep->GetPreStepPoint()->GetPosition().z() > -2117.5 ) )
	    { theStep->AddTotalEnergyDeposit(EnergyLoss); } // Disc 6
	  // Disc 7
	  else if ( (theStep->GetPreStepPoint()->GetPosition().z() > 2187.5 && theStep->GetPreStepPoint()->GetPosition().z() < 2307.5 )
		    || (theStep->GetPreStepPoint()->GetPosition().z() < -2187.5 && theStep->GetPreStepPoint()->GetPosition().z() > -2307.5 ) )
	    { theStep->AddTotalEnergyDeposit(EnergyLoss/(0.2)); } // Disc 7
	  // Disc 8
	  else if ( (theStep->GetPreStepPoint()->GetPosition().z() > 2392.5 && theStep->GetPreStepPoint()->GetPosition().z() < 2512.5 )
		    || (theStep->GetPreStepPoint()->GetPosition().z() < -2392.5 && theStep->GetPreStepPoint()->GetPosition().z() > -2512.5 ) )
	    { theStep->AddTotalEnergyDeposit(EnergyLoss/(0.2*0.2)); } // Disc 8
	  // Disc 9
	  else if ( (theStep->GetPreStepPoint()->GetPosition().z() > 2607.5 && theStep->GetPreStepPoint()->GetPosition().z() < 2727.5 )
		    || (theStep->GetPreStepPoint()->GetPosition().z() < -2607.5 && theStep->GetPreStepPoint()->GetPosition().z() > -2727.5 ) )
	    { theStep->AddTotalEnergyDeposit(EnergyLoss/(0.2*0.2*0.2)); } // Disc 9
	  // Beams in Barrel
	  else if ( (theStep->GetPreStepPoint()->GetPosition().z() > -870.0 && theStep->GetPreStepPoint()->GetPosition().z() < 1050.0) &&
		    (theStep->GetPreStepPoint()->GetPosition().perp() > 500.0 && theStep->GetPreStepPoint()->GetPosition().perp() < 630.0) )
	    { theStep->AddTotalEnergyDeposit(EnergyLoss/(0.2*0.2)); } // Beams in the Barrel
	  else
	    { 
	      // apparently we are not in a detector which should be hit by a LaserBeam
	      // therefore set the EnergyDeposit to zero and do not create a SimHit
	      theStep->ResetTotalEnergyDeposit(); 
	    }
	}
    }
  // kill the photon if it goes through a module in the outer barrel detector. In practice on the back 
  // of a module is a thin layer of Aluminium that absorbs the photons, so hits will only be created in 
  // the first layer of the TOB. In the current geometry this Aluminium layer is not included. This should
  // also avoid unwanted reflections (which then create hits in the TIB at the wrong positions)
  else if ( ( (theStep->GetPreStepPoint()->GetMaterial()->GetName() == "TOB_Wafer") &&
	      (theStep->GetPostStepPoint()->GetMaterial()->GetName() == "Air") ) )
    {
      LogDebug("LaserAlignmentSimulationStepping") << " AC1CMS: stepping aborted! ";
      theTrack->SetTrackStatus(fStopAndKill);
    }
  else if ( (theStep->GetPreStepPoint()->GetPosition().z() > -870.0 && 
	     theStep->GetPreStepPoint()->GetPosition().z() < 1050.0) &&
	    (theStep->GetPreStepPoint()->GetPosition().perp() > 630.0) )
    {
      LogDebug("LaserAlignmentSimulationStepping") << " AC1CMS: stepping aborted! ";
      theTrack->SetTrackStatus(fStopAndKill);
    }
  // do the same for photons that a) go through a module in the inner barrel detector or b) are reflected
  // at the surface of a TIB module. The photons in case b) can create hits in the TOB at the wrong z
  // positions :-(
  else if ( ( (theStep->GetPreStepPoint()->GetMaterial()->GetName() == "TIB_Wafer") &&
              (theStep->GetPostStepPoint()->GetMaterial()->GetName() == "Air") ) )
    {
      LogDebug("LaserAlignmentSimulationStepping") << " AC1CMS: stepping aborted! ";
      theTrack->SetTrackStatus(fStopAndKill);
    }
  else if ( (theStep->GetPreStepPoint()->GetPosition().z() > -870.0 && 
	     theStep->GetPreStepPoint()->GetPosition().z() < 1050.0) &&
	    (theStep->GetPreStepPoint()->GetPosition().perp() < 500.0) )
    {
      LogDebug("LaserAlignmentSimulationStepping") << " AC1CMS: stepping aborted! ";
      theTrack->SetTrackStatus(fStopAndKill);
    }
  // avoid reflections at Disc 1 of TEC- which enter again the Barrel Volume. These Photons
  // create hits at the wrong positions in TIB and TOB
  else if ( ( ( (theStep->GetPreStepPoint()->GetMaterial()->GetName() == "TEC_Wafer") &&
		(theStep->GetPostStepPoint()->GetMaterial()->GetName() == "T_Air")  ) ||
	      ( (theStep->GetPreStepPoint()->GetMaterial()->GetName() == "TEC_Wafer") &&
		(theStep->GetPostStepPoint()->GetMaterial()->GetName() == "Air") ) ) &&
	    (theStep->GetPreStepPoint()->GetMomentum().z() != theStep->GetPostStepPoint()->GetMomentum().z() ) &&
	    (theStep->GetPostStepPoint()->GetPosition().z() == -1137.25) )
    {
      LogDebug("LaserAlignmentSimulationStepping") << " AC1CMS: stepping aborted! photon in wrong direction";
      theTrack->SetTrackStatus(fStopAndKill);
    }
  // kill photons in the barrel which go in the wrong (i.e. +z) direction; they create unwanted hits
  // due to reflections ...
  else if ( ( theStep->GetPostStepPoint()->GetPosition().z() > -1100.0 )
	    && ( theStep->GetPostStepPoint()->GetPosition().z() < 1100.0 )
	    && ( theStep->GetPostStepPoint()->GetMomentumDirection().z() > 0.8 ) )
    {
      LogDebug("LaserAlignmentSimulationStepping") << " AC1CMS: stepping aborted! photon in wrong direction";
      theTrack->SetTrackStatus(fStopAndKill);
    }
  else
    {
      LogDebug("LaserAlignmentSimulationStepping") << " AC1CMS: stepping continuous ... ";
    }
  // ***********************************************************************************************************


  // check if it is alive
  if ( theTrack->GetTrackStatus() != fAlive ) 
    {
      LogDebug("LaserAlignmentSimulationStepping") << "<LaserSteppingAction::UserSteppingAction(const G4Step *)>: Track is not alive! -> return ";
      return; 
    }
  
  // check if it is a primary
  if ( theTrack->GetParentID() != 0 ) 
    { 
      LogDebug("LaserAlignmentSimulationStepping") << "<LaserSteppingAction::UserSteppingAction(const G4Step *)>: Track is not a primary! -> return ";
      return; 
    }

  // check if it is a optical photon
  if ( theDebugLevel >= 4 )
    {
      G4ParticleDefinition * theOpticalType = theTrack->GetDefinition();
      if ( theOpticalType == G4OpticalPhoton::OpticalPhotonDefinition() )
	{ 
	  LogDebug("LaserAlignmentSimulationStepping") << "<LaserSteppingAction::UserSteppingAction(const G4Step *)>: Optical Photon found! ";
	}

      // check in which volume it is
      G4StepPoint * thePreStepPoint = theStep->GetPreStepPoint();
      G4VPhysicalVolume * thePreStepPhysicalVolume = thePreStepPoint->GetPhysicalVolume();
      G4String thePreStepPhysicalVolumeName = thePreStepPhysicalVolume->GetName();
      G4Material * thePreStepMaterial = thePreStepPoint->GetMaterial();
      
      LogDebug("LaserAlignmentSimulationStepping") << "<LaserSteppingAction::UserSteppingAction(const G4Step *)>: the PreStep Position = " << thePreStepPoint->GetPosition()
					    << "\n<LaserSteppingAction::UserSteppingAction(const G4Step *)>: the PreStep Physical Volume = " << thePreStepPhysicalVolumeName
					    << "\n<LaserSteppingAction::UserSteppingAction(const G4Step *)>: the PreStep Material =" << thePreStepMaterial->GetName();

      G4StepPoint * thePostStepPoint = theStep->GetPostStepPoint();
      G4VPhysicalVolume * thePostStepPhysicalVolume = thePostStepPoint->GetPhysicalVolume();
      G4String thePostStepPhysicalVolumeName = thePostStepPhysicalVolume->GetName();
      G4Material * thePostStepMaterial = thePostStepPoint->GetMaterial();
      
      LogDebug("LaserAlignmentSimulationStepping") << "<LaserSteppingAction::UserSteppingAction(const G4Step *)>: the PostStep Position = " << thePostStepPoint->GetPosition()
					    << "\n<LaserSteppingAction::UserSteppingAction(const G4Step *)>: the PostStep Physical Volume = " << thePostStepPhysicalVolumeName
					    << "\n<LaserSteppingAction::UserSteppingAction(const G4Step *)>: the PostStep Material = " << thePostStepMaterial->GetName();
    }
}
