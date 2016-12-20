// Filter based on MCSingleParticleFilter.cc, but using rapidity instead of eta

#include "GeneratorInterface/GenFilters/interface/MCSingleParticleYPt.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


MCSingleParticleYPt::MCSingleParticleYPt(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared")))
{
   //here do whatever other initialization is needed
   vector<int> defpid ;
   defpid.push_back(0) ;
   particleID = iConfig.getUntrackedParameter< vector<int> >("ParticleID",defpid);  
   vector<double> defptmin ;
   defptmin.push_back(0.);
   ptMin = iConfig.getUntrackedParameter< vector<double> >("MinPt", defptmin);
   vector<double> defrapmin ;
   defrapmin.push_back(-10.);
   rapMin = iConfig.getUntrackedParameter< vector<double> >("MinY", defrapmin);
   vector<double> defrapmax ;
   defrapmax.push_back(10.);
   rapMax = iConfig.getUntrackedParameter< vector<double> >("MaxY", defrapmax);
   vector<int> defstat ;
   defstat.push_back(0);
   status = iConfig.getUntrackedParameter< vector<int> >("Status", defstat);

    // check for same size
    if ( (ptMin.size() > 1 &&  particleID.size() != ptMin.size()) 
     ||  (rapMin.size() > 1 && particleID.size() != rapMin.size()) 
     ||  (rapMax.size() > 1 && particleID.size() != rapMax.size())
     ||  (status.size() > 1 && particleID.size() != status.size()) ) {
      cout << "WARNING: MCSingleParticleYPt : size of vector cuts do not match!!" << endl;
    }

    // if ptMin size smaller than particleID , fill up further with defaults
    if (particleID.size() > ptMin.size() ){
       vector<double> defptmin2 ;
       for (unsigned int i = 0; i < particleID.size(); i++){ defptmin2.push_back(0.);}
       ptMin = defptmin2;   
    } 
    // if etaMin size smaller than particleID , fill up further with defaults
    if (particleID.size() > rapMin.size() ){
       vector<double> defrapmin2 ;
       for (unsigned int i = 0; i < particleID.size(); i++){ defrapmin2.push_back(-10.);}
       rapMin = defrapmin2;   
    } 
    // if etaMax size smaller than particleID , fill up further with defaults
    if (particleID.size() > rapMax.size() ){
       vector<double> defrapmax2 ;
       for (unsigned int i = 0; i < particleID.size(); i++){ defrapmax2.push_back(10.);}
       rapMax = defrapmax2;   
    }     
    // if status size smaller than particleID , fill up further with defaults
    if (particleID.size() > status.size() ){
       vector<int> defstat2 ;
       for (unsigned int i = 0; i < particleID.size(); i++){ defstat2.push_back(0);}
       status = defstat2;   
    } 
}


MCSingleParticleYPt::~MCSingleParticleYPt()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


// ------------ method called to skim the data  ------------
bool MCSingleParticleYPt::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   bool accepted = false;
   Handle<HepMCProduct> evt;
   iEvent.getByToken(token_, evt);

   const HepMC::GenEvent * myGenEvent = evt->GetEvent();
   for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();
	 p != myGenEvent->particles_end(); ++p ) {
     rapidity = 0.5*log( ((*p)->momentum().e()+(*p)->momentum().pz()) / ((*p)->momentum().e()-(*p)->momentum().pz()) );
     for (unsigned int i = 0; i < particleID.size(); i++) {
       if (particleID[i] == (*p)->pdg_id() || particleID[i] == 0) {
    
	 if ( (*p)->momentum().perp() > ptMin[i] 
              && rapidity > rapMin[i] && rapidity < rapMax[i] 
              && ((*p)->status() == status[i] || status[i] == 0) ) { 
           accepted = true;
           break;
	 }  
	 
       }
     }
     if (accepted) break;
   }
   
   if (accepted) { return true; } 
   else { return false; }
}

