#include "GeneratorInterface/Pythia6Interface/interface/TauolaInterface.h"

/*
 *  $Date: 2008/02/04 23:20:42 $
 *  $Revision: 1.4 $
 *  
 *  Christian Veelken
 *   04/17/07
 *
 */

#include <stdio.h>
#include <iostream>

#include <math.h>

// #include "CLHEP/HepMC/include/PythiaWrapper6_2.h"
// #include "GeneratorInterface/Pythia6Interface/interface/PythiaWrapper6_2.h"
#include "HepMC/PythiaWrapper6_2.h"
#include "GeneratorInterface/CommonInterface/interface/TauolaWrapper.h"

using namespace edm;

int TauolaInterface::debug_ = 0;
//int TauolaInterface::debug_ = 1;
//int TauolaInterface::debug_ = 2;

TauolaInterface::TauolaInterface()
{
//--- per default,
//    use TAUOLA package customized for CMS by Serge Slabospitsky 
  version_ = 1;

//--- per default,
//    enable polarization effects in tau lepton decays
  keypol_ = 1;
  
  // switch_photos_ = 0 ;
}

TauolaInterface::~TauolaInterface()
{}

void TauolaInterface::initialize()
{
//--- initialization of TAUOLA package;
//    to be called **before** the event loop

//--- check if TAUOLA tau decays are disabled
  if ( ki_taumod_.mdtau <= -1 ) return;

//--- disable tau decays in PYTHIA
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "Disabling tau decays in Pythia" << std::endl;
  int pdgCode = 15;
  int pythiaCode = call_pycomp(pdgCode);
  char parameter[20];
  sprintf(parameter, "MDCY(%i,1)=0", pythiaCode);
  std::cout << "pythiaCode = " << pythiaCode << std::endl;
  std::cout << "strlen(parameter) = " << strlen(parameter) << std::endl;
  call_pygive(parameter, strlen(parameter));
  std::cout << "----------------------------------------------" << std::endl;

//--- initialize TAUOLA
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "Initializing Tauola" << std::endl;
  int mode = -1;
  switch ( version_ ) {
  case 0 :
    call_tauola(mode, keypol_);
    break;
  case 1 :
    call_taurep(-2);
    // libra.ifphot = switch_photos_ ;
    call_tauola_srs(mode, keypol_);
    break;
  }
  std::cout << "----------------------------------------------" << std::endl;
}

void TauolaInterface::processEvent()
{
  if ( debug_ ) {
    std::cout << "<TauolaInterface::processEvent>: begin of event processing..." << std::endl;
  }

//--- convert PYJETS to HEPEVT common block structure
  if ( debug_ ) std::cout << "converting PYJETS to HEPEVT common block structure..." << std::endl;
  switch ( version_ ) {
  case 0 :
    call_pyhepc( 1 );
    break;
  case 1 : 
    call_pyhepc_t( 1 );
    break;
  }

//--- determine number of entries in HEPEVT common block **before** calling TAUOLA
  if ( debug_ ) std::cout << "determining number of generated particles **before** calling TAUOLA..." << std::endl;
  int dummy = -1;
  int numGenParticles_beforeTAUOLA = call_ihepdim(dummy);

//--- decay tau leptons with TAUOLA
  if ( debug_ ) std::cout << "calling TAUOLA..." << std::endl;
  int mode = 0;
  switch ( version_ ) {
  case 0 :
    call_tauola(mode, keypol_);
    break;
  case 1 : 
    call_tauola_srs(mode, keypol_);
    break;
  }
    
//--- determine number of entries in HEPEVT common block **after** calling TAUOLA
  if ( debug_ ) std::cout << "determining number of generated particles **after** calling TAUOLA..." << std::endl;
  //int dummy = -1;
  int numGenParticles_afterTAUOLA = call_ihepdim(dummy);


//--- convert back HEPEVT to PYJETS common block structure
  if ( debug_ ) std::cout << "converting back HEPEVT to PYJETS common block structure..." << std::endl;
  switch ( version_ ) {
  case 0 :
    call_pyhepc( 2 );
    break;
  case 1 : 
    call_pyhepc_t( 2 );
    break;
  }

//--- simulated further decay of unstable hadrons 
//    produced in tau decay
  if ( debug_ ) std::cout << "decaying unstable hadrons produced in tau decay..." << std::endl;

  int numGenParticles_afterUnstableHadronDecays = decayUnstableHadrons(numGenParticles_beforeTAUOLA, numGenParticles_afterTAUOLA);


//--- set production vertex for tau decay products,
//    taking into account tau lifetime of c tau = 87 um
  if ( debug_ ) std::cout << "setting decay vertex of tau lepton and production vertices of decay products..." << std::endl;
  //setDecayVertex(numGenParticles_beforeTAUOLA, numGenParticles_afterTAUOLA);
  setDecayVertex(numGenParticles_beforeTAUOLA, numGenParticles_afterUnstableHadronDecays);

//--- simulated further decay of unstable hadrons 
//    produced in tau decay
//  if ( debug_ ) std::cout << "decaying unstable hadrons produced in tau decay..." << std::endl;
//  decayUnstableHadrons(numGenParticles_beforeTAUOLA, numGenParticles_afterTAUOLA);


//--- print list of particles
  if ( debug_ > 1 ) {
    int nFirst = numGenParticles_beforeTAUOLA;
    int nLast = pyjets.n;
    for ( int iParticle = (nFirst + 1); iParticle <= nLast; ++iParticle ) {
      std::cout << "genParticle #" << iParticle << std::endl;
      std::cout << " type = " 
		<< pyjets.k[1][iParticle - 1] << std::endl;
      std::cout << " p(Px,Py,Pz,E) = {" 
		<< pyjets.p[0][iParticle - 1] << ","
		<< pyjets.p[1][iParticle - 1] << ","
		<< pyjets.p[2][iParticle - 1] << ","
		<< pyjets.p[3][iParticle - 1] << "}" << std::endl;
      std::cout << " v(x,y,z) = {" 
		<< pyjets.v[0][iParticle - 1] << ","
		<< pyjets.v[1][iParticle - 1] << ","
		<< pyjets.v[2][iParticle - 1] << "}" << std::endl;
    }
  }

  if ( debug_ ) {
    std::cout << "<TauolaInterface::processEvent>: end of event processing..." << std::endl;
  }
}

void TauolaInterface::print()
{
//--- print event generation statistics
//    and branching fraction information of TAUOLA
  int mode = 1;
  switch ( version_ ) {
  case 0 :
    call_tauola(mode, keypol_);
    break;
  case 1 : 
    call_tauola_srs(mode, keypol_);
    break;
  }
}

void TauolaInterface::setDecayVertex(int numGenParticles_beforeTAUOLA, int numGenParticles_afterTAUOLA)
{
//-------------------------------------------------------------------------------
//               set production vertex for tau decay products,
//             taking tau lifetime of c tau = 87 um into account
//-------------------------------------------------------------------------------

  int numDocumentationLines = pypars.msti[3];

  if ( debug_ > 1 ) {
    std::cout << "numGenParticles_beforeTAUOLA = " << numGenParticles_beforeTAUOLA << std::endl;
    std::cout << "numGenParticles_afterTAUOLA = " << numGenParticles_afterTAUOLA << std::endl;
    std::cout << "numDocumentationLines = " << numDocumentationLines << std::endl;
  }

//--- check that index given for first particle resulting from tau decay is valid
//    and in particular does not point into documentation lines at beginning of PYJETS common block
  if ( numGenParticles_beforeTAUOLA <= numDocumentationLines ) {
    std::cerr << "Error in <TauolaInterface::setDecayVertex>: index of first particle in PYJETS common block less than number of documentation lines !" << std::endl;
    return;
  }

//--- temporary storage of particle decay vertices;
//    reset for each event
  double particleDecayVertex[4][maxNumberParticles];
  for ( int i = 0; i < 4; ++i ) {
    for ( int iParticle = 1; iParticle <= maxNumberParticles; ++iParticle ) {
      particleDecayVertex[i][iParticle] = 0.;
    }
  }

  int nFirst = numDocumentationLines + 1;
  int nLast = numGenParticles_afterTAUOLA;

  for ( int iParticle = nFirst; iParticle <= nLast; ++iParticle ) {
    int particleStatus = pyjets.k[0][iParticle - 1];
    if ( debug_ > 1 ) {
      std::cout << "genParticle[" << iParticle << "] status = " << particleStatus << std::endl;
    }

//--- check that particle is "currently existing"
//    and not e.g. a documentation line	summarizing event level information
//    if ( particleStatus > 0 &&
//	   particleStatus < 10 ) {
    if ( particleStatus == 1 ||
	 particleStatus == 11 ) {
      int particleType = abs(pyjets.k[1][iParticle - 1]);
      int particleType_PYTHIA = call_pycomp(particleType);
      
      if ( debug_ > 1 ) {
	std::cout << "genParticle[" << iParticle << "] type = " << particleType << std::endl;
      }

//--- do not touch particles not associated to tau decays
      if ( particleType != 15 && iParticle <= numGenParticles_beforeTAUOLA ) continue;

//--- set production vertex 
//    for daughter particles resulting from tau decay
//    to decay vertex of their mother
      for ( int i = 0; i < 4; ++i ) {
	if ( particleType != 15 ) {
	  int indexParticleMother_i = pyjets.k[2][iParticle - 1];
	  
	  if ( indexParticleMother_i > iParticle ) {
	    std::cerr << "Error in <TauolaInterface::setDecayVertex>: daughter found before mother particle in PYJETS common block !" << std::endl;
	  } 
	  
	  if ( debug_ > 1 ) {
	    if ( i == 0 ) std::cout << " --> setting production vertex for genParticle[" << iParticle << "]" 
				    << " to decay vertex of genParticle[" << indexParticleMother_i << "]" << std::endl;
	  }

	  double particleProductionVertex_i = particleDecayVertex[i][indexParticleMother_i - 1];
	  pyjets.v[i][iParticle - 1] = particleProductionVertex_i;
	}
      }

//-------------------------------------------------------------------------------
//  for unstable particles, randomly choose lifetime and determine decay vertex 
//-------------------------------------------------------------------------------

      if ( particleStatus == 11 ) {

//--- set time after which particle decayed in its restframe
//    (prob = exp(-t/lifetime) ==> t = -lifetime * log(prob),
//     with prob randomly chosen from flat distribution between zero and one)
	double lifetime = pydat2.pmas[3][particleType_PYTHIA - 1];
	double u = call_pyr(0); // random number generated by PYTHIA
	double ct = -lifetime*log(u); // time (in particle rest frame) after which particle decayed
	pyjets.v[4][iParticle - 1] = ct;
	
	if ( debug_ > 1 ) {
	  std::cout << "lifetime = " << lifetime << std::endl;
	  std::cout << "u = " << u << std::endl;
	  std::cout << "ct = " << ct << std::endl;
	}

//--- set decay vertex
//    (first three coordinates = x,y,z;
//     fourth coordinate = time after which particle decayed in laboratory frame;
//     see e.g.
//       http://cepa.fnal.gov/psm/simulation/mcgen/lund/pythia_manual/pythia6.3/pythia6301/node35.html
//     for description of the PYTHIA common block structures)
//
	for ( int i = 0; i < 4; ++i ) {
	  double particleProductionVertex_i = pyjets.v[i][iParticle - 1];

//--- set decay vertex = production vertex + ct*v*gamma,
//    with 
//      v = unit vector in particle direction 
//    and 
//      gamma = E/m Lorentz boost factor
//	    
	  double particleMomentum_i = pyjets.p[i][iParticle - 1];
	  double particleEnergy = pyjets.p[3][iParticle - 1];
	  double particleMass = pyjets.p[4][iParticle - 1];
	  if ( particleEnergy >= particleMass && particleMass > 0. ) {
	    double particleMomentum = sqrt(particleEnergy*particleEnergy - particleMass*particleMass);
	    
	    double particleDecayVertex_i = particleProductionVertex_i + ct*(particleMomentum_i/particleMomentum)*(particleEnergy/particleMass);
	    
	    if ( debug_ > 1 ) {
	      if ( i == 0 ) {
		std::cout << "particleEnergy = " << particleEnergy << std::endl;
		std::cout << "particleMass = " << particleMass << std::endl;
		std::cout << "particleMomentum = " << particleMomentum << std::endl;
	      }
	      std::cout << "particleMomentum_" << i << " = " << particleMomentum_i << std::endl;
	      if ( i == 0 ) std::cout << " --> setting decay vertex of genParticle[" << iParticle << "]" << std::endl;
	      std::cout << "particleDecayVertex[" << i << "] = " << particleDecayVertex_i << std::endl;
	    }
	    
	    particleDecayVertex[i][iParticle - 1] = particleDecayVertex_i;
	  } else {
	    if ( particleMass == 0 ) 
	      std::cerr << "Error in <TauolaInterface::setDecayVertex>: mass of unstable particle cannot be zero !" << std::endl;
	    else
	      std::cerr << "Error in <TauolaInterface::setDecayVertex>: energy of tau lepton cannot be smaller than its mass !" << std::endl;
	    continue;
	  }
	}
      }
    } 
  }
}

int TauolaInterface::decayUnstableHadrons(int numGenParticles_beforeTAUOLA, int numGenParticles_afterTAUOLA)
{
//-------------------------------------------------------------------------------
//          further decay of unstable hadrons produced in tau decay
//-------------------------------------------------------------------------------

//--- check if all particles resulting from tau decay are stable; 
//    keep iterating as long as n12 > n11 
//    (some decay products are unstable)
  int nFirst = numGenParticles_beforeTAUOLA + 1;
  int nLast = numGenParticles_afterTAUOLA;
  while ( nLast > nFirst && nLast < maxNumberParticles ) {
    if ( debug_ > 1 ) {
      std::cout << "before calling PYDECY:" << std::endl;
      std::cout << " nFirst = " << nFirst << std::endl;
      std::cout << " nLast = " << nLast << std::endl;
    }

    for ( int iParticle = nFirst; iParticle <= nLast; ++iParticle ) {
      int particleStatus = pyjets.k[0][iParticle - 1];
      if ( debug_ > 1 ) {
	std::cout << "genParticle[" << iParticle << "] status = " << particleStatus << std::endl;
      }

//--- check that particle is "currently existing"
//    and not e.g. a documentation line	summarizing event level information
      if ( particleStatus > 0 &&
	   particleStatus < 10 ) {
	int particleType = abs(pyjets.k[1][iParticle - 1]);
	
	if ( debug_ > 1 ) {
	  std::cout << "genParticle[" << iParticle << "] type = " << particleType << std::endl;
	}

	if ( particleType >= 100 ) {
	  if ( particleType != 211 &&   // PI+- 
	       particleType != 321 &&   // K+- 
	       particleType != 130 &&   // KL 
	       particleType != 310 ) {  // KS
	    if ( debug_ > 1 ) std::cout << "calling PYDECY for genParticle[" << iParticle << "]" << std::endl;
	    call_pydecy(iParticle);
	  }
	}
      }
    }

    nFirst = nLast + 1;
    nLast = pyjets.n;

    if ( debug_ > 1 ) {
      std::cout << "after calling PYDECY:" << std::endl;
      std::cout << " nFirst = " << nFirst << std::endl;
      std::cout << " nLast = " << nLast << std::endl;
    }

//--- mark particles that did not decay within the CMS detector volume
//    as undecayed
    for ( int iParticle = nFirst; iParticle <= nLast; ++iParticle ) {
      int particleStatus = pyjets.k[0][iParticle - 1];
      int particleType = abs(pyjets.k[1][iParticle - 1]);
      int particleDaughters = pyjets.k[3][iParticle - 1];
      if ( particleStatus == 4 &&
	   particleType >= 100 &&
	   particleDaughters == 0 ) {
	if ( debug_ > 1 ) std::cout << "setting status = 1 for genParticle[" << iParticle << "]" << std::endl;
	pyjets.k[0][iParticle - 1] = 1;
      }
    }
  }

//--- return last index of particle resulting from tau decay
//    (including direct and indirect decay products)
  return nLast;
}
