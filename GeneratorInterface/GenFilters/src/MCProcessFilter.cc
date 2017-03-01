
#include "GeneratorInterface/GenFilters/interface/MCProcessFilter.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

using namespace edm;
using namespace std;


MCProcessFilter::MCProcessFilter(const edm::ParameterSet& iConfig) :
token_(consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared")))
{
   //here do whatever other initialization is needed
   vector<int> defproc ;
   defproc.push_back(0) ;
   processID = iConfig.getUntrackedParameter< vector<int> >("ProcessID",defproc);  
   vector<double> defpthatmin ;
   defpthatmin.push_back(0.);
   pthatMin = iConfig.getUntrackedParameter< vector<double> >("MinPthat", defpthatmin);
   vector<double> defpthatmax ;
   defpthatmax.push_back(10000.);
   pthatMax = iConfig.getUntrackedParameter< vector<double> >("MaxPthat", defpthatmax);


    // checkin size of phthat vectors -- default is allowed
    if ( (pthatMin.size() > 1 &&  processID.size() != pthatMin.size()) 
     ||  (pthatMax.size() > 1 && processID.size() != pthatMax.size()) ) {
      cout << "WARNING: MCPROCESSFILTER : size of MinPthat and/or MaxPthat not matching with ProcessID size!!" << endl;
    }

    // if pthatMin size smaller than processID , fill up further with defaults 
    if (processID.size() > pthatMin.size() ){ 
       vector<double> defpthatmin2 ;
       for (unsigned int i = 0; i < processID.size(); i++){ defpthatmin2.push_back(0.);}
       pthatMin = defpthatmin2;
    }     
    // if pthatMax size smaller than processID , fill up further with defaults 
    if (processID.size() > pthatMax.size() ){
       vector<double> defpthatmax2 ;
       for (unsigned int i = 0; i < processID.size(); i++){ defpthatmax2.push_back(10000.);}
       pthatMax = defpthatmax2;   
    }


}


MCProcessFilter::~MCProcessFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


// ------------ method called to skim the data  ------------
bool MCProcessFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   bool accepted = false;
   Handle<HepMCProduct> evt;
   iEvent.getByToken(token_, evt);

   const HepMC::GenEvent * myGenEvent = evt->GetEvent();
   
   
   // do the selection -- processID 0 is always accepted
   for (unsigned int i = 0; i < processID.size(); i++){
     if (processID[i] == myGenEvent->signal_process_id() || processID[i] == 0) {
       
       if ( myGenEvent->event_scale() > pthatMin[i] &&  myGenEvent->event_scale() < pthatMax[i] ) { 
	 accepted = true; 
       }  
       
     } 
   }
   
   if (accepted){ return true; } else {return false;}
   
}

