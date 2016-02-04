/** \file LaserPrimaryGeneratorAction.cc
 *  
 *
 *  $Date: 2007/06/11 14:44:29 $
 *  $Revision: 1.3 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignmentSimulation/interface/LaserPrimaryGeneratorAction.h"
#include "SimG4Core/Notification/interface/GenParticleInfo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

LaserPrimaryGeneratorAction::LaserPrimaryGeneratorAction(edm::ParameterSet const& theConf) 
  : thePhotonEnergy(0),
    thenParticleInGun(0),
    thenParticle(0),
    theLaserBeamsInTEC1(),
    theLaserBeamsInTEC2(),
    theLaserBeamsInTECTIBTOBTEC()
{
  // {{{ LaserPrimaryGeneratorAction constructor

  // get the PhotonEnergy from the parameter set
  thePhotonEnergy = theConf.getUntrackedParameter<double>("PhotonEnergy",1.15) * eV;

  // number of particles in the Laser beam
  thenParticleInGun = theConf.getUntrackedParameter<int>("NumberOfPhotonsInParticleGun",1);

  // number of particles in one beam. ATTENTION: each beam contains nParticleInGun with the same
  // startpoint and direction. nParticle gives the number of particles in the beam with a different
  // startpoint. They are used to simulate the gaussian beamprofile of the Laser Beams.
  thenParticle = theConf.getUntrackedParameter<int>("NumberOfPhotonsInEachBeam",1);

  // create a messenger for this class
//   theGunMessenger = new LaserPrimaryGeneratorMessenger(this);
  
  // create the beams in the right endcap
  theLaserBeamsInTEC1 = new LaserBeamsTEC1(thenParticleInGun, thenParticle, thePhotonEnergy);

  // create the beams in the left endcap
  theLaserBeamsInTEC2 = new LaserBeamsTEC2(thenParticleInGun, thenParticle, thePhotonEnergy);

  // create the beams to connect the TECs with TOB and TIB
  theLaserBeamsInTECTIBTOBTEC = new LaserBeamsBarrel(thenParticleInGun, thenParticle, thePhotonEnergy);
  // }}}
}

LaserPrimaryGeneratorAction::~LaserPrimaryGeneratorAction()
{
  // {{{ LaserPrimaryGeneratorAction destructor

  if ( theLaserBeamsInTEC1 != 0 ) { delete theLaserBeamsInTEC1; }
  if ( theLaserBeamsInTEC2 != 0 ) { delete theLaserBeamsInTEC2; }
  if ( theLaserBeamsInTECTIBTOBTEC != 0 ) { delete theLaserBeamsInTECTIBTOBTEC; }
  // }}}
}

void LaserPrimaryGeneratorAction::GeneratePrimaries(G4Event* myEvent)
{
  // {{{ GeneratePrimaries (G4Event * myEvent)

  // this function is called at the beginning of an Event in LaserAlignment::upDate(const BeginOfEvent * myEvent)
  LogDebug("LaserPrimaryGeneratorAction") << "<LaserPrimaryGeneratorAction::GeneratePrimaries(G4Event*)>: create a new Laser Event";

  // shoot in the right endcap
  theLaserBeamsInTEC1->GeneratePrimaries(myEvent);

  // shoot in the left endcap
  theLaserBeamsInTEC2->GeneratePrimaries(myEvent);

  // shoot in the barrel
  theLaserBeamsInTECTIBTOBTEC->GeneratePrimaries(myEvent);

  // loop over all the generated vertices, get the primaries and set the user information
  int theID = 0;

  for (int i = 1; i < myEvent->GetNumberOfPrimaryVertex(); i++)
    {
      G4PrimaryVertex * theVertex = myEvent->GetPrimaryVertex(i);

      for (int j = 0; j < theVertex->GetNumberOfParticle(); j++)
	{
	  G4PrimaryParticle * thePrimary = theVertex->GetPrimary(j);
	  
	  setGeneratorId(thePrimary, theID);
	  theID++;
	}
    }
  // }}}
}

void LaserPrimaryGeneratorAction::setGeneratorId(G4PrimaryParticle * aParticle, int ID) const
{
  // {{{ SetGeneratorId(G4PrimaryParticle * aParticle, int ID) const

  /* *********************************************************************** */
  /*   OSCAR expacts each G4PrimaryParticle to have some User Information    *
   *   therefore this function have been implemented                         */
  /* *********************************************************************** */

  aParticle->SetUserInformation(new GenParticleInfo(ID));
  // }}}
}
