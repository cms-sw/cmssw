
#include "GeneratorInterface/GenFilters/interface/MCSmartSingleParticleFilter.h"
#include "GeneratorInterface/GenFilters/interface/MCFilterZboostHelper.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


MCSmartSingleParticleFilter::MCSmartSingleParticleFilter(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared"))),
betaBoost(iConfig.getUntrackedParameter("BetaBoost",0.))
{
   //here do whatever other initialization is needed
   vector<int> defpid ;
   defpid.push_back(0) ;
   particleID = iConfig.getUntrackedParameter< vector<int> >("ParticleID",defpid);  
   vector<double> defpmin ;
   defpmin.push_back(0.);
   pMin = iConfig.getUntrackedParameter< vector<double> >("MinP", defpmin);

   vector<double> defptmin ;
   defptmin.push_back(0.);
   ptMin = iConfig.getUntrackedParameter< vector<double> >("MinPt", defptmin);

   vector<double> defetamin ;
   defetamin.push_back(-10.);
   etaMin = iConfig.getUntrackedParameter< vector<double> >("MinEta", defetamin);
   vector<double> defetamax ;
   defetamax.push_back(10.);
   etaMax = iConfig.getUntrackedParameter< vector<double> >("MaxEta", defetamax);
   vector<int> defstat ;
   defstat.push_back(0);
   status = iConfig.getUntrackedParameter< vector<int> >("Status", defstat);

   vector<double> defDecayRadiusmin;
   defDecayRadiusmin.push_back(-1.);
   decayRadiusMin = iConfig.getUntrackedParameter< vector<double> >("MinDecayRadius", defDecayRadiusmin);

   vector<double> defDecayRadiusmax;
   defDecayRadiusmax.push_back(1.e5);
   decayRadiusMax = iConfig.getUntrackedParameter< vector<double> >("MaxDecayRadius", defDecayRadiusmax);

   vector<double> defDecayZmin;
   defDecayZmin.push_back(-1.e5);
   decayZMin = iConfig.getUntrackedParameter< vector<double> >("MinDecayZ", defDecayZmin);

   vector<double> defDecayZmax;
   defDecayZmax.push_back(1.e5);
   decayZMax = iConfig.getUntrackedParameter< vector<double> >("MaxDecayZ", defDecayZmax);

    // check for same size
    if ( (pMin.size() > 1   && particleID.size() != pMin.size())
     ||  (ptMin.size() > 1  && particleID.size() != ptMin.size())
     ||  (etaMin.size() > 1 && particleID.size() != etaMin.size())
     ||  (etaMax.size() > 1 && particleID.size() != etaMax.size())
     ||  (status.size() > 1 && particleID.size() != status.size())
     ||  (decayRadiusMin.size() > 1 && particleID.size() != decayRadiusMin.size())
     ||  (decayRadiusMax.size() > 1 && particleID.size() != decayRadiusMax.size())
     ||  (decayZMin.size() > 1 && particleID.size() != decayZMin.size())
     ||  (decayZMax.size() > 1 && particleID.size() != decayZMax.size()) ) {
      cout << "WARNING: MCPROCESSFILTER : size of MinPthat and/or MaxPthat not matching with ProcessID size!!" << endl;
    }

    // if pMin size smaller than particleID , fill up further with defaults
    if (particleID.size() > pMin.size() ){
       for (unsigned int i = pMin.size(); i < particleID.size(); i++){ pMin.push_back(0.);}
    } 
    // if ptMin size smaller than particleID , fill up further with defaults
    if (particleID.size() > ptMin.size() ){
       for (unsigned int i = ptMin.size(); i < particleID.size(); i++){ ptMin.push_back(0.);}
    } 
    // if etaMin size smaller than particleID , fill up further with defaults
    if (particleID.size() > etaMin.size() ){
       for (unsigned int i = etaMin.size(); i < particleID.size(); i++){ etaMin.push_back(-10.);}
    } 
    // if etaMax size smaller than particleID , fill up further with defaults
    if (particleID.size() > etaMax.size() ){
       for (unsigned int i = etaMax.size(); i < particleID.size(); i++){ etaMax.push_back(10.);}
    }
    // if status size smaller than particleID , fill up further with defaults
    if (particleID.size() > status.size() ){
       for (unsigned int i = status.size(); i < particleID.size(); i++){ status.push_back(0);}
    }

    // if decayRadiusMin size smaller than particleID , fill up further with defaults
    if (particleID.size() > decayRadiusMin.size() ){
       for (unsigned int i = decayRadiusMin.size(); i < particleID.size(); i++){ decayRadiusMin.push_back(-10.);}
    }
    // if decayRadiusMax size smaller than particleID , fill up further with defaults
    if (particleID.size() > decayRadiusMax.size() ){
       for (unsigned int i = decayRadiusMax.size(); i < particleID.size(); i++){ decayRadiusMax.push_back(1.e5);}
    }

    // if decayZMin size smaller than particleID , fill up further with defaults
    if (particleID.size() > decayZMin.size() ){
       for (unsigned int i = decayZMin.size(); i < particleID.size(); i++){ decayZMin.push_back(-1.e5);}
    }
    // if decayZMax size smaller than particleID , fill up further with defaults
    if (particleID.size() > decayZMax.size() ){
       for (unsigned int i = decayZMax.size(); i < particleID.size(); i++){ decayZMax.push_back(1.e5);}
    }

    // check if beta is smaller than 1
    if (std::abs(betaBoost) >= 1 ){
      edm::LogError("MCSmartSingleParticleFilter") << "Input beta boost is >= 1 !";
    }

}


MCSmartSingleParticleFilter::~MCSmartSingleParticleFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


// ------------ method called to skim the data  ------------
bool MCSmartSingleParticleFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   bool accepted = false;
   Handle<HepMCProduct> evt;
   iEvent.getByToken(token_, evt);

   const HepMC::GenEvent * myGenEvent = evt->GetEvent();
     
   
   for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	 p != myGenEvent->particles_end(); ++p ) {

    
     for (unsigned int i = 0; i < particleID.size(); i++){
       if (particleID[i] == (*p)->pdg_id() || particleID[i] == 0) {

	 if ( (*p)->momentum().perp() > ptMin[i]
              && ((*p)->status() == status[i] || status[i] == 0))  {

           HepMC::FourVector mom = MCFilterZboostHelper::zboost((*p)->momentum(),betaBoost);
           if ( mom.rho() > pMin[i]
                && mom.eta() > etaMin[i] && mom.eta() < etaMax[i]) {

             if (!((*p)->production_vertex())) continue;
	   
             double decx = (*p)->production_vertex()->position().x();
             double decy = (*p)->production_vertex()->position().y();
             double decrad = sqrt(decx*decx+decy*decy);
             if (decrad<decayRadiusMin[i] ) continue;
             if (decrad>decayRadiusMax[i] ) continue;

             double decz = (*p)->production_vertex()->position().z();
             if (decz<decayZMin[i] ) continue;
             if (decz>decayZMax[i] ) continue;

             accepted = true;
           }

         }
	 
       } 
     }
     
     
   }

   if (accepted){ return true; } else {return false;}

}
