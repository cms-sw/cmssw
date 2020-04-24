// livio.fano@cern.ch

#include "GeneratorInterface/GenFilters/interface/BHFilter.h"
#include "DataFormats/Common/interface/Handle.h"

using namespace std;
namespace cms

{
BHFilter::BHFilter(const edm::ParameterSet& conf):    conf_(conf)
{
  rBounds = conf_.getParameter< vector<double> >("radii");
  zBounds = conf_.getParameter< vector<double> >("zeds");
  bFields = conf_.getParameter< vector<double> >("bfiel");
  bReduction = conf_.getParameter< double >("factor");
  trig_ = conf_.getParameter< int >("trig_type");
  trig2_ = conf_.getParameter< int >("scintillators_type");

  for ( unsigned i=0; i<bFields.size(); ++i ) { 
    bFields[i] *= bReduction;
    cout << "r/z/b = " << rBounds[i] << " " << zBounds[i] << " " << bFields[i] << endl;
  }

  if(trig_==0){ cout <<endl << "trigger is both + and - BSC " << endl;}
  if(trig_==1){ cout <<endl << "trigger is + side BSC " << endl;}
  if(trig_==-1){ cout <<endl << "trigger is - side BSC " << endl;}
  
  cout << endl;
  
  if(trig2_==0){ cout <<endl << "trigger is both PADs and DISKs " << endl;}
  if(trig2_==-1){ cout <<endl << "trigger is only PADs " << endl;}
  if(trig2_==1){ cout <<endl << "trigger is only DISKs " << endl;}

  if(trig_!=0 && trig_!=-1 && trig_!=1 && trig2_!=0 && trig2_!=-1 && trig2_!=1)
    {
      cout <<endl << endl <<endl<< "WARNING!! BSC trigger/scintillator type not properly defined " << endl;
      cout <<endl << endl <<endl<< "WARNING!! BSC trigger/scintillator type not properly defined " << endl << endl << endl;
    }

}

bool BHFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{


  edm::Handle<edm::HepMCProduct>HepMCEvt;
  iEvent.getByLabel("generator","unsmeared",HepMCEvt);
  const HepMC::GenEvent* MCEvt = HepMCEvt->GetEvent();

  BaseParticlePropagator PP;
  map<unsigned,XYZTLorentzVector> myHits;



  for(HepMC::GenEvent::particle_const_iterator i=MCEvt->particles_begin(); i != MCEvt->particles_end();++i)
    {



      pad_plus = false;
      pad_minus = false;
      circ_plus = false;
      circ_minus = false;

      int myId = (*i)->pdg_id();


	const HepMC::GenVertex * vertex_=(*i)->production_vertex();
	double xv = vertex_->position().x();
	double yv = vertex_->position().y();
	double zv = vertex_->position().z();
	double tv = vertex_->position().t();
	XYZTLorentzVector vertex = XYZTLorentzVector(xv,yv,zv,tv);

	//vertex_->position();

	//	HepMC::FourVector 
	XYZTLorentzVector momentum = XYZTLorentzVector(((*i)->momentum()).x(), ((*i)->momentum()).y(), ((*i)->momentum()).z(), ((*i)->momentum()).t());

	RawParticle myMuon(-momentum, vertex/10.);

	  if ( myId < 0 ) 
	    myMuon.setCharge(-1.);
	  else
	    myMuon.setCharge(+1.);

	  BaseParticlePropagator PP(myMuon,0.,0.,0.);
	  

	  float x0= vertex.x();
	  float y0= vertex.y();
	  float z0= vertex.z();
	  float px0=momentum.x();
	  float py0=momentum.y();
	  float pz0=momentum.z();

	  // beam 1 or 2 ?
	  // propagator -> need to be implemented 
	  

	      // intersection point - particle/BSC1
	      float zs = 10860.;
	      float ys = (zs-z0)*(py0/pz0)+y0;
	      float xs = (ys-y0)*(px0/py0)+x0;
	      
	      
	      if(xs<0 && ys>0) {xs=-xs;}
	      if(xs>0 && ys<0) {ys=-ys;}
	      if(xs<0 && ys<0) {xs=-xs; ys=-ys;}
	      
	      
	      // scintillator pads
	      float A[2]={732.7, 24.7};
	      float B[2]={895.3, 187.3};
	      float C[2]={144.0, 850.2};
	      float D[2]={69.8, 776.0};
	      float m1=(B[1]-A[1])/(B[0]-A[0]);
	      float m2=(C[1]-B[1])/(C[0]-B[0]);
	      float m3=(D[1]-C[1])/(D[0]-C[0]);
	      float m4=(A[1]-D[1])/(A[0]-D[0]);
	      float y1=m1*(xs-A[0])+A[1];
	      float y2=m2*(xs-B[0])+B[1];
	      float y3=m3*(xs-C[0])+C[1];
	      float y4=m4*(xs-D[0])+D[1];
	      
	      
	      
	      
	      // trigger conditions
	      
	      if(ys>y1 && ys<y2 && ys<y3 && ys>y4)
		{
		  pad_plus = true;
		  //  cout << " trig1+" << endl;
		}
	      
	      if((ys<sqrt(450*450-xs*xs) && ys>sqrt(208*208-xs*xs)) || (ys<sqrt(450*450-xs*xs) && xs>208)) 
		{
		  circ_plus = true;
		  //  cout << " trig2+" << endl;
		}
	      
	      
	      // intersection point - particle/BSC2
	      zs = -10860.;
	      ys = (zs-z0)*(py0/pz0)+y0;
	      xs = (ys-y0)*(px0/py0)+x0;
	      
	      //  cout << endl << " xs ys zs " << xs << " " << ys << " " << zs; 
	      
	      
	      if(xs<0 && ys>0) {xs=-xs;}
	      if(xs>0 && ys<0) {ys=-ys;}
	      if(xs<0 && ys<0) {xs=-xs; ys=-ys;}
	      
	      
	      // scintillator pads
	      y1=m1*(xs-A[0])+A[1];
	      y2=m2*(xs-B[0])+B[1];
	      y3=m3*(xs-C[0])+C[1];
	      y4=m4*(xs-D[0])+D[1];
	      
	      
	      
	      // trigger conditions
	      
	      if(ys>y1 && ys<y2 && ys<y3 && ys>y4)
		{
		  pad_minus = true;
		  // cout << " trig1-" << endl;
		}
	      
	      if((ys<sqrt(450*450-xs*xs) && ys>sqrt(208*208-xs*xs)) || (ys<sqrt(450*450-xs*xs) && xs>208)) 
		{
		  circ_minus = true;
		  //   cout << " trig2-" << endl;
		}
	      
	      
	      // final selection

	      if(trig2_==-1) 
		{
		  pad_plus = false;
		  pad_minus = false;
		}
	      if(trig2_==1) 
		{
		  circ_plus = false;
		  circ_minus = false; 
		}
	      
	      if(trig_==0 && (pad_plus || circ_plus) && (pad_minus || circ_minus) )
		{
		  //		  cout << "triggg 0 " << endl;
		  return true;
		}
	      if(trig_==1 && (pad_plus || circ_plus)) 
		{
		  //		  cout << "triggg 1 " << endl;
		  return true;
		}
	      if(trig_==-1 && (pad_minus || circ_minus)) 
		{
		  //		  cout << "triggg -1 " << endl;
		  return true;
		}

    }
	  
	  return false;

}



}
