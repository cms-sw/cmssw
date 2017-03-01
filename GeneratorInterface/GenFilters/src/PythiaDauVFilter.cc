#include "GeneratorInterface/GenFilters/interface/PythiaDauVFilter.h"


#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/PythiaWrapper6_4.h"
#include <iostream>
#include <vector>

using namespace edm;
using namespace std;


PythiaDauVFilter::PythiaDauVFilter(const edm::ParameterSet& iConfig) :
  fVerbose(iConfig.getUntrackedParameter("verbose",0)),
  token_(consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared"))),
  particleID(iConfig.getUntrackedParameter("ParticleID", 0)),
  motherID(iConfig.getUntrackedParameter("MotherID", 0)),
  chargeconju(iConfig.getUntrackedParameter("ChargeConjugation", true)),
  ndaughters(iConfig.getUntrackedParameter("NumberDaughters", 0)),
  //minptcut(iConfig.getUntrackedParameter("MinPt", 0.)),
  maxptcut(iConfig.getUntrackedParameter("MaxPt", 14000.))
  //minetacut(iConfig.getUntrackedParameter("MinEta", -10.)),
  //maxetacut(iConfig.getUntrackedParameter("MaxEta", 10.)) 
{
  //now do what ever initialization is needed
  vector<int> defdauID;
  defdauID.push_back(0);
  dauIDs = iConfig.getUntrackedParameter< vector<int> >("DaughterIDs",defdauID);
  vector<double> defminptcut;
  defminptcut.push_back(0.);
  minptcut = iConfig.getUntrackedParameter< vector<double> >("MinPt",defminptcut);
  vector<double> defminetacut;
  defminetacut.push_back(-10.);
  minetacut = iConfig.getUntrackedParameter< vector<double> >("MinEta",defminetacut);
  vector<double> defmaxetacut;
  defmaxetacut.push_back(10.);
  maxetacut = iConfig.getUntrackedParameter< vector<double> >("MaxEta",defmaxetacut);

  cout << "----------------------------------------------------------------------" << endl;
  cout << "--- PythiaDauVFilter" << endl;
  for (unsigned int i=0; i<dauIDs.size(); ++i) {
    cout << "ID: " <<  dauIDs[i] << " pT > " << minptcut[i] << " " << minetacut[i] << " eta < " << maxetacut[i] << endl;
  }
  cout << "maxptcut   = " << maxptcut << endl;
  cout << "particleID = " << particleID << endl;
  cout << "motherID   = " << motherID << endl;
  cout << "----------------------------------------------------------------------" << endl;

}


PythiaDauVFilter::~PythiaDauVFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
bool PythiaDauVFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  bool accepted = false;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);

  int OK(1); 
  vector<int> vparticles; 
  
  HepMC::GenEvent *myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));
  
  if (fVerbose > 5) {
    cout << "looking for " << particleID << endl;
  }
    
  for (HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p) {
    
    if ((*p)->pdg_id() != particleID) continue ;

    // -- Check for mother of this particle
    if (0 != motherID) {
      OK = 0; 
      for (HepMC::GenVertex::particles_in_const_iterator des = (*p)->production_vertex()->particles_in_const_begin(); 
	   des != (*p)->production_vertex()->particles_in_const_end();
	   ++des) {
	if (fVerbose > 10) {
	  cout << "mother: " << (*des)->pdg_id() << " pT: " << (*des)->momentum().perp() << " eta: " << (*des)->momentum().eta() << endl;
	}
	if (abs(motherID) == abs((*des)->pdg_id())) {
	  OK = 1; 
	  break;
	}
      }
    }
    if (0 == OK) continue; 

    // -- check for daugthers
    int ndauac = 0;
    int ndau = 0;     
    if (fVerbose > 5) {
      cout << "found ID: " << (*p)->pdg_id() << " pT: " << (*p)->momentum().perp() << " eta: " << (*p)->momentum().eta() << endl;
    }
    vparticles.push_back((*p)->pdg_id()); 
    if ((*p)->end_vertex()) {	
      for (HepMC::GenVertex::particle_iterator des=(*p)->end_vertex()->particles_begin(HepMC::children);
	   des != (*p)->end_vertex()->particles_end(HepMC::children);
	   ++des) {
	++ndau;       
	if (fVerbose > 5) {
	  cout << "ID: " << (*des)->pdg_id() << " pT: " << (*des)->momentum().perp() << " eta: " << (*des)->momentum().eta() << endl;
	}
	for (unsigned int i=0; i<dauIDs.size(); ++i) {
	  if ((*des)->pdg_id() != dauIDs[i] ) continue ;
	  if (fVerbose > 5) {
	    cout << "i = " << i << " pT = " << (*des)->momentum().perp() << " eta = " << (*des)->momentum().eta() << endl;
	  }
	  if ((*des)->momentum().perp() >  minptcut[i]  &&
	      (*des)->momentum().perp() <  maxptcut  &&
	      (*des)->momentum().eta()  >  minetacut[i] && 
	      (*des)->momentum().eta()  <  maxetacut[i] ) {
	    ++ndauac;
	    vparticles.push_back((*des)->pdg_id()); 
	    if (fVerbose > 2) {
	      cout << "  accepted this particle " <<  (*des)->pdg_id()
		   << " pT = " << (*des)->momentum().perp() << " eta = " << (*des)->momentum().eta() << endl;
	    }
	    break;
	  } 
	}	       		     
      }
    }  

    // -- allow photons
    if (ndau >=  ndaughters && ndauac == ndaughters) {
      accepted = true;
      if (fVerbose > 0) {
	cout << "  accepted this decay: ";
	for (unsigned int iv = 0; iv < vparticles.size(); ++iv) cout << vparticles[iv] << " "; 
	cout << " from mother = " << motherID << endl;
      }
      break;
    }    
    
  }
  
  
  if (!accepted && chargeconju ) {
    OK = 1; 

    for (HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
	 p != myGenEvent->particles_end(); ++p) {
      
      if ((*p)->pdg_id() != -particleID) continue ;
      
      // -- Check for mother of this particle
      if (0 != motherID) {
	OK = 0; 
	for (HepMC::GenVertex::particles_in_const_iterator des = (*p)->production_vertex()->particles_in_const_begin(); 
	     des != (*p)->production_vertex()->particles_in_const_end();
	     ++des) {
	  if (fVerbose > 10) {
	    cout << "mother: " << (*des)->pdg_id() << " pT: " << (*des)->momentum().perp() << " eta: " << (*des)->momentum().eta() << endl;
	  }
	  if (abs(motherID) == abs((*des)->pdg_id())) {
	    OK = 1; 
	    break;
	  }
	}
      }
      if (0 == OK) continue; 
      
      if (fVerbose > 5) {
	cout << "found ID: " << (*p)->pdg_id() << " pT: " << (*p)->momentum().perp() << " eta: " << (*p)->momentum().eta() << endl;
      }
      vparticles.push_back((*p)->pdg_id()); 
      int ndauac = 0;
      int ndau = 0;     
      if ((*p)->end_vertex()) {
	for (HepMC::GenVertex::particle_iterator des=(*p)->end_vertex()->particles_begin(HepMC::children);
	     des != (*p)->end_vertex()->particles_end(HepMC::children);
	     ++des) {
	  ++ndau;
	  if (fVerbose > 5) {
	    cout << "ID: " << (*des)->pdg_id() << " pT: " << (*des)->momentum().perp() << " eta: " << (*des)->momentum().eta() << endl;
	  }
	  for (unsigned int i=0; i<dauIDs.size(); ++i) {
	    int IDanti = -dauIDs[i];
	    int pythiaCode = PYCOMP(dauIDs[i]);
	    int has_antipart = pydat2.kchg[3-1][pythiaCode-1];
	    if (has_antipart == 0) IDanti = dauIDs[i];
	    if ((*des)->pdg_id() != IDanti) continue ;
	    if (fVerbose > 5) {
	      cout << "i = " << i << " pT = " << (*des)->momentum().perp() << " eta = " << (*des)->momentum().eta() << endl;
	    }
	    if ((*des)->momentum().perp() >  minptcut[i]  &&
		(*des)->momentum().perp() <  maxptcut  &&
		(*des)->momentum().eta()  >  minetacut[i] && 
		(*des)->momentum().eta()  <  maxetacut[i] ) {
	      ++ndauac;
	      vparticles.push_back((*des)->pdg_id()); 
	      if (fVerbose > 2) {
		cout << "  accepted this particle " <<  (*des)->pdg_id()
		     << " pT = " << (*des)->momentum().perp() << " eta = " << (*des)->momentum().eta() << endl;
	      }
	      break;
	    } 
	  }	       		     
	}
      }
      if (ndau >=  ndaughters && ndauac == ndaughters ) {
	accepted = true;
	if (fVerbose > 0) {
	  cout << "  accepted this anti-decay: ";
	  for (unsigned int iv = 0; iv < vparticles.size(); ++iv) cout << vparticles[iv] << " "; 
	  cout << " from mother = " << motherID << endl;
	}
	break;
      }    
    }
    
  }    
  
  delete myGenEvent; 
  
  
  if (accepted){
    return true; 
  } else {
    return false;
  }
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(PythiaDauVFilter);
