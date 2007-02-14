//Framework Headers
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/src/VectorInputSourceFactory.h"

#include "DataFormats/Common/interface/ModuleDescription.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"

#include "FastSimulation/MaterialEffects/interface/NuclearInteractionEDMUpdator.h"
#include "FastSimulation/MaterialEffects/interface/NUEvent.h"

#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include <iostream>
#include <cmath>
#include "TFile.h"
#include "TTree.h"

using namespace std;

NuclearInteractionEDMUpdator::NuclearInteractionEDMUpdator(const edm::ParameterSet& inputFiles,
							   std::vector<double>& pionEnergies,
							   double pionEnergy,
							   double lengthRatio) :
  MaterialEffectsUpdator(),
  input(edm::VectorInputSourceFactory::get()->makeVectorInputSource(
		inputFiles,edm::InputSourceDescription()).release()),
  md_(),
  thePionCM(pionEnergies),
  thePionEnergy(pionEnergy),
  theLengthRatio(lengthRatio)

{

  // Compute the corresponding cm energies of the nuclear interactions
  HepLorentzVector Proton(0.,0.,0.,0.986);
  for ( unsigned file=0; file<thePionCM.size(); ++file ) {

    double thePionMass2 = 0.13957*0.13957;
    double thePionMomentum2 = thePionCM[file]*thePionCM[file] - thePionMass2;
    HepLorentzVector Reference(0.,0.,sqrt(thePionMomentum2),thePionCM[file]);
    thePionCM[file] = (Reference+Proton).mag();

  }


}

NuclearInteractionEDMUpdator::~NuclearInteractionEDMUpdator() {}

void NuclearInteractionEDMUpdator::compute(ParticlePropagator& Particle)
{

  // Read a Nuclear Interaction in a random manner
  using namespace edm; 

  double eHadron = Particle.momentum().vect().mag(); 

  // The hadron has enough energy to create some relevant final state
  if ( eHadron > thePionEnergy ) { 

    // Probability to interact is dl/L0
    if ( -log(random->flatShoot()) <= radLengths * theLengthRatio ) {

      HepLorentzVector Proton(0.,0.,0.,0.986);
      HepLorentzVector Hadron(Particle.momentum());
      double ecm = (Proton+Hadron).mag();
      // Get the file of interest (closest c.m. energy)
      unsigned file = 0;
      double dmin = 1E8;
      for ( ; file<thePionCM.size(); ++file ) {
	double dist = fabs ( ecm/thePionCM[file] - 1. );
	//	std::cout << "file/dist = " << file << " " << dist << std::endl;
	if ( dist < dmin )  
	  dmin = dist;
	else 
	  break;
      }
      --file;

      // The boost characteristics
      HepLorentzVector theBoost = Proton + Hadron;
      theBoost /= theBoost.e();
      
      // Get the nuclear interaction of interest
      EventPrincipalVector result;
      edm::Handle<std::vector<SimTrack> > mySimTracks;
      int entry = (int) (file *1E8) + (int) (random->flatShoot() * 1E8);
      input->readMany(entry,result); // Warning! we read here only one entry !
      Event e(**(result.begin()),md_);
      e.getByLabel("prodNU",mySimTracks);
      
      unsigned nTracks = mySimTracks->size();
      for ( unsigned iTrack=0; iTrack<nTracks; ++iTrack ) { 
	
	// Create a RawParticle with the proper energy in the c.m frame of 
	// the nuclear interaction
	RawParticle * myPart = new  RawParticle ((*mySimTracks)[iTrack].type(),
						 (*mySimTracks)[iTrack].momentum()*ecm);
	// Boost it in the lab frame
	myPart->boost(theBoost.x(),theBoost.y(),theBoost.z());
	
	// Update the daughter list
	_theUpdatedState.push_back(myPart);

      }

    }

  }

}

