/** \file LaserBeamsTEC1.cc
 *  
 *
 *  $Date: 2011/09/16 06:40:53 $
 *  $Revision: 1.8 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignmentSimulation/interface/LaserBeamsTEC1.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "globals.hh"                        // Global Constants and typedefs
#include "G4ParticleDefinition.hh"
#include "G4ParticleGun.hh"

LaserBeamsTEC1::LaserBeamsTEC1() :
  theParticleGun(0),
  theDRand48Engine(0)
{
  G4int nPhotonsGun = 1;
  G4int nPhotonsBeam = 1;
  G4double Energy = 1.15 * eV;
  // call constructor with options
  LaserBeamsTEC1(nPhotonsGun, nPhotonsBeam, Energy);
}

LaserBeamsTEC1::LaserBeamsTEC1(G4int nPhotonsInGun, G4int nPhotonsInBeam, G4double PhotonEnergy) : thenParticleInGun(0),
												   thenParticle(0),
												   thePhotonEnergy(0),
												   theParticleGun(),
												   theDRand48Engine()
{
  /* *********************************************************************** */
  /*  initialize and configure the particle gun                              */
  /* *********************************************************************** */

  // the Photon energy
  thePhotonEnergy = PhotonEnergy;

  // number of particles in the Laser beam
  thenParticleInGun = nPhotonsInGun;

  // number of particles in one beam. ATTENTION: each beam contains nParticleInGun with the same
  // startpoint and direction. nParticle gives the number of particles in the beam with a different
  // startpoint. They are used to simulate the gaussian beamprofile of the Laser Beams.
  thenParticle = nPhotonsInBeam;

  // create the particle gun
  theParticleGun = new G4ParticleGun(thenParticleInGun);

  // default kinematics
  G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
  G4ParticleDefinition * theOpticalPhoton = theParticleTable->FindParticle("opticalphoton");

  theParticleGun->SetParticleDefinition(theOpticalPhoton);
  theParticleGun->SetParticleTime(0.0 * ns);
  theParticleGun->SetParticlePosition(G4ThreeVector(-500.0 * cm, 0.0 * cm, 0.0 * cm));
  theParticleGun->SetParticleMomentumDirection(G4ThreeVector(5.0, 3.0, 0.0));
  theParticleGun->SetParticleEnergy(10.0 * keV);
  setOptPhotonPolar(90.0);

  // initialize the random number engine
  theDRand48Engine = new CLHEP::DRand48Engine();

}

LaserBeamsTEC1::~LaserBeamsTEC1()
{
  if ( theParticleGun != 0 )  { delete theParticleGun; }
  if ( theDRand48Engine != 0 ) { delete theDRand48Engine; }
}

void LaserBeamsTEC1::GeneratePrimaries(G4Event* myEvent)
{
  // this function is called at the beginning of an Event in LaserAlignment::upDate(const BeginOfEvent * myEvent)

  // use the random number generator service of the framework
  edm::Service<edm::RandomNumberGenerator> rng;
  unsigned int seed = rng->mySeed();

  // set the seed
  theDRand48Engine->setSeed(seed);

  // number of LaserRings and Laserdiodes
  const G4int nLaserRings = 2;
  const G4int nLaserBeams = 8;

  // z position of the sixth Tracker Endcap Disc, where the Laserdiodes are positioned
  G4double LaserPositionZ = 2057.5 * mm;

  // Radius of the inner and outer Laser ring
  G4double LaserRingRadius[nLaserRings] = {564.0 * mm, 840.0 * mm};

  // phi position of the first Laserdiode
  G4double LaserPhi0 = 0.392699082;

  // width of the LaserBeams
  G4double LaserRingSigmaX[nLaserRings] = {0.5 * mm, 0.5 * mm};
  G4double LaserRingSigmaY[nLaserRings] = {0.5 * mm, 0.5 * mm};

  // get the definition of the optical photon
  G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
  G4ParticleDefinition * theOpticalPhoton= theParticleTable->FindParticle("opticalphoton");
  
  // loop over the LaserRings
  for (int theRing = 0; theRing < nLaserRings; theRing++)
    {
      // loop over the LaserBeams
      for (int theBeam = 0; theBeam < nLaserBeams; theBeam++)
	{
	  // code for forward and backward beam
	  // calculate the right phi for the current beam
	  G4double LaserPositionPhi = LaserPhi0 + G4double(theBeam * G4double(G4double(2 * M_PI) / nLaserBeams));

	  // calculate x and y position for the current beam
	  G4double LaserPositionX = cos(LaserPositionPhi) * LaserRingRadius[theRing];
	  G4double LaserPositionY = sin(LaserPositionPhi) * LaserRingRadius[theRing];

	  // loop over all the particles in one beam
	  for (int theParticle = 0; theParticle < thenParticle; theParticle++)
	    {
	      // get randomnumbers  and calculate the position
	      CLHEP::RandGaussQ aGaussObjX( *theDRand48Engine, LaserPositionX, LaserRingSigmaX[theRing] );
	      CLHEP::RandGaussQ aGaussObjY( *theDRand48Engine, LaserPositionY, LaserRingSigmaY[theRing] );

	      G4double theXPosition = aGaussObjX.fire();
	      G4double theYPosition = aGaussObjY.fire();
	      G4double theZPosition = LaserPositionZ;

	      // set the properties of the newly created particle
	      theParticleGun->SetParticleDefinition(theOpticalPhoton);
	      theParticleGun->SetParticleTime(0.0 * ns);
	      theParticleGun->SetParticlePosition(G4ThreeVector(theXPosition, theYPosition, theZPosition));
	      theParticleGun->SetParticleEnergy(thePhotonEnergy);

	      // loop over both directions of the beam
	      for (int theDirection = 0; theDirection < 2; theDirection++)
		{
		  // shoot in both beam directions ...
		  if (theDirection == 0)  // shoot in forward direction (+z)
		    {
		      theParticleGun->SetParticleMomentumDirection(G4ThreeVector(0.0, 0.0, 1.0));
		      // set the polarization
		      setOptPhotonPolar(90.0);
		      // generate the particle
		      theParticleGun->GeneratePrimaryVertex(myEvent);
		    }

		  if (theDirection == 1)  // shoot in backward direction (-z)
		    {
		      theParticleGun->SetParticleMomentumDirection(G4ThreeVector(0.0, 0.0, -1.0));
		      // set the polarization
		      setOptPhotonPolar(90.0);
		      // generate the particle
		      theParticleGun->GeneratePrimaryVertex(myEvent);
		    }
		} // end loop over both beam directions
	    } // end loop over particles in beam
	} // end loop over beams
    } // end loop over rings
}

void LaserBeamsTEC1::setOptPhotonPolar(G4double Angle)
{
  /* *********************************************************************** */
  /*   to get optical processes working properly, you have to make sure      *
   *   that the photon polarisation is defined.                              */
  /* *********************************************************************** */

  // first check if we have an optical photon
  if ( theParticleGun->GetParticleDefinition()->GetParticleName() != "opticalphoton" )
    { 
      edm::LogWarning("SimLaserAlignment:LaserBeamsTEC1") << "<LaserBeamsTEC1::SetOptPhotonPolar()>: WARNING! The ParticleGun is not an optical photon";
      return;
    }

//   G4cout << "  AC1CMS: The ParticleGun is an " << theParticleGun->GetParticleDefinition()->GetParticleName();
  G4ThreeVector normal(1.0, 0.0, 0.0);
  G4ThreeVector kphoton = theParticleGun->GetParticleMomentumDirection();
  G4ThreeVector product = normal.cross(kphoton);
  G4double modul2 = product * product;

  G4ThreeVector e_perpendicular(0.0, 0.0, 1.0);
  
  if ( modul2 > 0.0 ) { e_perpendicular = (1.0 / sqrt(modul2)) * product; }
  
  G4ThreeVector e_parallel = e_perpendicular.cross(kphoton);

  G4ThreeVector polar = cos(Angle) * e_parallel + sin(Angle) * e_perpendicular;
  
//   G4cout << ", the polarization = " << polar << G4endl;
  theParticleGun->SetParticlePolarization(polar);
}

