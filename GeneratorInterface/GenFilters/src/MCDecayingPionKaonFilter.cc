#include "GeneratorInterface/GenFilters/interface/MCDecayingPionKaonFilter.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


MCDecayingPionKaonFilter::MCDecayingPionKaonFilter(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter("moduleLabel",std::string("generator"))))
{
   //here do whatever other initialization is needed
   vector<int> defpid ;
   defpid.push_back(0) ;
   particleID = iConfig.getUntrackedParameter< vector<int> >("ParticleID",defpid);  
   vector<double> defptmin ;
   defptmin.push_back(0.);
   ptMin = iConfig.getUntrackedParameter< vector<double> >("MinPt", defptmin);

   vector<double> defetamin ;
   defetamin.push_back(-10.);
   etaMin = iConfig.getUntrackedParameter< vector<double> >("MinEta", defetamin);
   vector<double> defetamax ;
   defetamax.push_back(10.);
   etaMax = iConfig.getUntrackedParameter< vector<double> >("MaxEta", defetamax);

   vector<double> defDecayRadiusmin ;
   defDecayRadiusmin.push_back(-10.);
   decayRadiusMin = iConfig.getUntrackedParameter< vector<double> >("MinDecayRadius", defDecayRadiusmin);

   vector<double> defDecayRadiusmax ;
   defDecayRadiusmax.push_back(1.e5);
   decayRadiusMax = iConfig.getUntrackedParameter< vector<double> >("MaxDecayRadius", defDecayRadiusmax);

   vector<double> defDecayZmin ;
   defDecayZmin.push_back(-1.e5);
   decayZMin = iConfig.getUntrackedParameter< vector<double> >("MinDecayZ", defDecayZmin);

   vector<double> defDecayZmax ;
   defDecayZmax.push_back(1.e5);
   decayZMax = iConfig.getUntrackedParameter< vector<double> >("MaxDecayZ", defDecayZmax);

    ptMuMin = iConfig.getUntrackedParameter<double>("PtMuMin",0.);  
    // check for same size
    if ( (ptMin.size() > 1 &&  particleID.size() != ptMin.size())
     ||  (etaMin.size() > 1 && particleID.size() != etaMin.size())
     ||  (etaMax.size() > 1 && particleID.size() != etaMax.size())
     ||  (decayRadiusMin.size() > 1 && particleID.size() != decayRadiusMin.size())
     ||  (decayRadiusMax.size() > 1 && particleID.size() != decayRadiusMax.size())
     ||  (decayZMin.size() > 1 && particleID.size() != decayZMin.size())
     ||  (decayZMax.size() > 1 && particleID.size() != decayZMax.size()) ) {
      cout << "WARNING: MCPROCESSFILTER : size of MinPthat and/or MaxPthat not matching with ProcessID size!!" << endl;
    }

    // if ptMin size smaller than particleID , fill up further with defaults
    if (particleID.size() > ptMin.size() ){
       vector<double> defptmin2 ;
       for (unsigned int i = 0; i < particleID.size(); i++){ defptmin2.push_back(0.);}
       ptMin = defptmin2;   
    } 
    // if etaMin size smaller than particleID , fill up further with defaults
    if (particleID.size() > etaMin.size() ){
       vector<double> defetamin2 ;
       for (unsigned int i = 0; i < particleID.size(); i++){ defetamin2.push_back(-10.);}
       etaMin = defetamin2;   
    } 
    // if etaMax size smaller than particleID , fill up further with defaults
    if (particleID.size() > etaMax.size() ){
       vector<double> defetamax2 ;
       for (unsigned int i = 0; i < particleID.size(); i++){ defetamax2.push_back(10.);}
       etaMax = defetamax2;   
    }     

    // if decayRadiusMin size smaller than particleID , fill up further with defaults
    if (particleID.size() > decayRadiusMin.size() ){
       vector<double> decayRadiusmin2 ;
       for (unsigned int i = 0; i < particleID.size(); i++){ decayRadiusmin2.push_back(-10.);}
       decayRadiusMin = decayRadiusmin2;   
    } 
    // if decayRadiusMax size smaller than particleID , fill up further with defaults
    if (particleID.size() > decayRadiusMax.size() ){
       vector<double> decayRadiusmax2 ;
       for (unsigned int i = 0; i < particleID.size(); i++){ decayRadiusmax2.push_back(1.e5);}
       decayRadiusMax = decayRadiusmax2;   
    }     

    // if decayZMin size smaller than particleID , fill up further with defaults
    if (particleID.size() > decayZMin.size() ){
       vector<double> decayZmin2 ;
       for (unsigned int i = 0; i < particleID.size(); i++){ decayZmin2.push_back(-1.e5);}
       decayZMin = decayZmin2;   
    } 
    // if decayZMax size smaller than particleID , fill up further with defaults
    if (particleID.size() > decayZMax.size() ){
       vector<double> decayZmax2 ;
       for (unsigned int i = 0; i < particleID.size(); i++){ decayZmax2.push_back(1.e5);}
       decayZMax = decayZmax2;   
    }     

}


MCDecayingPionKaonFilter::~MCDecayingPionKaonFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


// ------------ method called to skim the data  ------------
bool MCDecayingPionKaonFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   bool accepted = false;
   Handle<HepMCProduct> evt;
   iEvent.getByToken(token_, evt);

   const HepMC::GenEvent * myGenEvent = evt->GetEvent();
     
   
    for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	  p != myGenEvent->particles_end(); ++p ) {

    
     for (unsigned int i = 0; i < particleID.size(); i++){
      if (!((*p)->end_vertex())) continue;
      if (particleID[i] != (*p)->pdg_id() && particleID[i] != 0) continue;

      if ( (*p)->momentum().perp() < ptMin[i] ) continue;

      if ( (*p)->momentum().eta() < etaMin[i] ) continue;
      if ( (*p)->momentum().eta() > etaMax[i] ) continue;

      double decx = (*p)->end_vertex()->position().x();
      double decy = (*p)->end_vertex()->position().y();
      double decrad = sqrt(decx*decx+decy*decy);
      if (decrad<decayRadiusMin[i] ) continue;
      if (decrad>decayRadiusMax[i] ) continue;

      double decz = (*p)->end_vertex()->position().z();
      if (decz<decayZMin[i] ) continue;
      if (decz>decayZMax[i] ) continue;


      if ((*p)->status()==2) {
            for (HepMC::GenVertex::particle_iterator         
              vpdec= (*p)->end_vertex()->particles_begin(HepMC::children);
              vpdec != (*p)->end_vertex()->particles_end(HepMC::children); 
              ++vpdec) {
                  if (abs((*vpdec)->pdg_id())!=13) continue;
                  if (fabs((*vpdec)->momentum().perp())>ptMuMin) {
                        accepted = true;
                        break;
                  }
            }
      } else if ((*p)->status()==1) {
            accepted = true;
      }

     } 


    }

    delete myGenEvent; 


   if (accepted){ return true; } else {return false;}

}

