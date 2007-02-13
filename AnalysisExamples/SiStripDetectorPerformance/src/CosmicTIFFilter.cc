// livio.fano@cern.ch

#include "AnalysisExamples/SiStripDetectorPerformance/interface/CosmicTIFFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"
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
CosmicTIFFilter::CosmicTIFFilter(const edm::ParameterSet& conf):    conf_(conf)
{
/*
  rBounds = conf_.getParameter< vector<double> >("radii");
  zBounds = conf_.getParameter< vector<double> >("zeds");
  bFields = conf_.getParameter< vector<double> >("bfiel");
  bReduction = conf_.getParameter< double >("factor");

  for ( unsigned i=0; i<bFields.size(); ++i ) { 
    bFields[i] *= bReduction;
    //    cout << "r/z/b = " << rBounds[i] << " " << zBounds[i] << " " << bFields[i] << endl;
  }
*/
}

bool CosmicTIFFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


  edm::Handle<edm::HepMCProduct>HepMCEvt;
  iEvent.getByLabel("source","",HepMCEvt);
  const HepMC::GenEvent* MCEvt = HepMCEvt->GetEvent();


/*
  BaseParticlePropagator PP;
  map<unsigned,HepLorentzVector> myHits;
*/

  bool hit1=false;
  bool hit2=false;
  bool hit3=false;


  for(HepMC::GenEvent::particle_const_iterator i=MCEvt->particles_begin(); i != MCEvt->particles_end();++i)
    {

      int myId = (*i)->ParticleID();
      if (abs(myId)==13)
      {

	// Get the muon position and momentum
	HepLorentzVector vertex=(*i)->CreationVertex();
	CLHEP::HepLorentzVector momentum=(*i)->Momentum();


	// Define the Scintillator position
	HepLorentzVector S1(0,3000.,500);
	HepLorentzVector S2(0,-1500.,500);
	HepLorentzVector S3(0,3000.,3500);
	
	hit1=Sci_trig(vertex, momentum, S1);
	hit2=Sci_trig(vertex, momentum, S2);
	hit3=Sci_trig(vertex, momentum, S3);
	
	// trigger conditions

	if(hit1)cout << "1 " << hit1 << endl;
	if(hit2)cout << "2 " << hit2 << endl;
	if(hit3)cout << "3 " << hit3 << endl;


	if((hit1&&hit2) || (hit3&&hit2))
	  {
	    cout << " got a trig " << endl; 
	    return true;
	  }
      }
    }
  


  return false;
}

bool CosmicTIFFilter::Sci_trig(HepLorentzVector vertex,  HepLorentzVector momentum, HepLorentzVector S)
{
	  float x0= vertex.x();
	  float y0= vertex.y();
	  float z0= vertex.z();
	  float px0=momentum.x();
	  float py0=momentum.y();
	  float pz0=momentum.z();
	  float Sx=S.x();
	  float Sy=S.y();
	  float Sz=S.z();

	  //float ys=Sy;
	  float zs=(Sy-y0)*(pz0/py0)+z0;
	  //	  float xs=((Sy-y0)*(pz0/py0)-z0)*(px0/pz0)+x0;
	  float xs=(Sy-y0)*(px0/py0)+x0;

	  //	  cout << Sx << " " << Sz << " " << xs << " " << zs << endl;
	  //	  cout << x0 << " " << z0 << " " << px0 << " " << py0 << " " << pz0 << endl;

	  if((xs<Sx+500 && xs>Sx-500)&&(zs<Sz+500 && zs>Sz-500))
	    {
	      return true;
	    }
	  else
	    {
	      return false;
	    }

}

}

