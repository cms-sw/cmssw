// patrick.janot@cern.ch, livio.fano@cern.ch

#include "AnalysisExamples/SiStripDetectorPerformance/interface/CosmicGenFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMap.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <map>
#include <vector>

using namespace std;
namespace cms

{
CosmicGenFilter::CosmicGenFilter(const edm::ParameterSet& conf):    conf_(conf)
{

  rBounds = conf_.getParameter< vector<double> >("radii");
  zBounds = conf_.getParameter< vector<double> >("zeds");
  bFields = conf_.getParameter< vector<double> >("bfiel");
  bReduction = conf_.getParameter< double >("factor");

  for ( unsigned i=0; i<bFields.size(); ++i ) { 
    bFields[i] *= bReduction;
    cout << "r/z/b = " << rBounds[i] << " " << zBounds[i] << " " << bFields[i] << endl;
  }
}

bool CosmicGenFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // Step A: Get Inputs
//  std::cout << "*************************** theStripHits.size()   = " << theStripHits.size()   << std::endl;

  edm::Handle<edm::HepMCProduct>HepMCEvt;
  iEvent.getByLabel("source","",HepMCEvt);
  const HepMC::GenEvent* MCEvt = HepMCEvt->GetEvent();

  BaseParticlePropagator PP;
  map<unsigned,HepLorentzVector> myHits;

  for(HepMC::GenEvent::particle_const_iterator i=MCEvt->particles_begin(); i != MCEvt->particles_end();++i)
    {

      int myId = (*i)->ParticleID();
      if (abs(myId)==13)
      {

	// Get the muon position and momentum
	  HepLorentzVector vertex=(*i)->CreationVertex();
	  CLHEP::HepLorentzVector momentum=(*i)->Momentum();

	  // Set-up (back) propagation -> momentum and charge are reversed
	  RawParticle myMuon(-momentum, vertex/10.);
	  if ( myId < 0 ) 
	    myMuon.setCharge(-1.);
	  else
	    myMuon.setCharge(+1.);

	  BaseParticlePropagator PP(myMuon,0.,0.,0.);
	  
	  // Propagate in magnetic field
	  unsigned i = 0;
	  bool test = true;
	  int success = 1;
	  while ( i<rBounds.size() && test && success > 0 ) {
	    // Set the magnetic field to the value corresponding to that "layer"
	    PP.setMagneticField(bFields[i]);
	    // First propagate to the closest approach of the z axis,
	    // so as to go "inside" the cylinder
	    // (This is only because the propagate() method works 
	    // from inside to outside
	    test = PP.propagateToClosestApproach();
	    // Second propagate from the distance of closest approach 
	    // to the cylinder in question
	    PP.setPropagationConditions(rBounds[i], zBounds[i], false);
	    test = PP.propagate();
	    // If the track cannot meet the cylinder, the test value is false
	    if ( test ) { 
	      // Success = 1 (Barrel) or 2 (Endcap) 
	      success = PP.getSuccess();
	      if ( success > 0 ) myHits[rBounds.size()-i] = PP.vertex();
	      ++i;
	    }
	  }

      }
      // verify if condition are satisfied

      int nValidHits = 0;
      for ( unsigned i=1; i<5; ++i ) { 
	if ( myHits.find(i) != myHits.end() ) {
	  HepLorentzVector& theHit = myHits[i];

		cout <<" zz " <<  theHit.z() << endl;

		cout <<" phi " <<  theHit.phi() << endl;
		cout << " ph2 " << atan2(theHit.y(),theHit.x()) << endl;

	  if ( theHit.z() > 0. && 
	       theHit.z() < 120. && 

	       //fix the phi in a more reasonable way: tob r=~100 tob width ~20cm (exagerating) 
	
	       theHit.phi() > 0.8 && 
	       theHit.phi() < 2.4 ) ++nValidHits;
	}
      }

      if( nValidHits >= 3 ) { 
	cout << "Event ACCEPTED !" << endl;
	// Check hits
	for ( unsigned i=1; i<rBounds.size()+1; ++i )
	  if ( myHits.find(i) != myHits.end() ) 
	    cout << "Found vertex : " << i << " " 
		 << myHits[i].x() << " " 
		 << myHits[i].y() << " " 
		 << myHits[i].z() << endl; 
	return true;
      }

    }

  return false;
}

}
