// livio.fano@cern.ch

#include "GeneratorInterface/GenFilters/interface/CosmicTIFFilter.h"
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
CosmicTIFFilter::CosmicTIFFilter(const edm::ParameterSet& conf):    conf_(conf)
{

  Sci1 = conf_.getParameter< vector <double>  >("scintillator1");
  Sci2 = conf_.getParameter< vector <double>  >("scintillator2");
  Sci3 = conf_.getParameter< vector <double>  >("scintillator3");
  cout << " scintillators positions: " << endl;
  cout << " S1 x " << Sci1[0] << " y " << Sci1[1] << " z " << Sci1[2] << endl;
  cout << " S2 x " << Sci2[0] << " y " << Sci2[1] << " z " << Sci2[2] << endl;
  cout << " S3 x " << Sci3[0] << " y " << Sci3[1] << " z " << Sci3[2] << endl;

}

bool CosmicTIFFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


  edm::Handle<edm::HepMCProduct>HepMCEvt;
  iEvent.getByLabel("source","",HepMCEvt);
  const HepMC::GenEvent* MCEvt = HepMCEvt->GetEvent();


  bool hit1=false;
  bool hit2=false;
  bool hit3=false;


  for(HepMC::GenEvent::particle_const_iterator i=MCEvt->particles_begin(); i != MCEvt->particles_end();++i)
    {

      //      int myId = (*i)->ParticleID();

      // New HepMC
      int myId = (*i)->pdg_id();


      if (abs(myId)==13)
      {

	// Get the muon position and momentum
	//HepLorentzVector vertex=(*i)->CreationVertex();

	// new HepMC
	const HepMC::GenVertex * vertex_=(*i)->production_vertex();

	//	HepMC::FourVector vertex__ = vertex_->position();
	CLHEP::HepLorentzVector vertex = ((vertex_->position()).x(),(vertex_->position()).y(),(vertex_->position()).z());

	//HepMC::FourVector momentum__=(*i)->momentum();
	CLHEP::HepLorentzVector momentum=(((*i)->momentum()).x(),((*i)->momentum()).y(),((*i)->momentum()).z(),((*i)->momentum()).t());


	// Define the Scintillator position

	HepLorentzVector S1(Sci1[0],Sci1[1],Sci1[2]);
	HepLorentzVector S2(Sci2[0],Sci2[1],Sci2[2]);
	HepLorentzVector S3(Sci3[0],Sci3[1],Sci3[2]);

	hit1=Sci_trig(vertex, momentum, S1);
	hit2=Sci_trig(vertex, momentum, S2);
	hit3=Sci_trig(vertex, momentum, S3);


	// trigger conditions


	if((hit1&&hit2) || (hit3&&hit2))
	  {
	    cout << " got a trig " << endl; 
	    if(hit1)cout << "1 " << hit1 << endl;
	    if(hit2)cout << "2 " << hit2 << endl;
	    if(hit3)cout << "3 " << hit3 << endl;
	    
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

