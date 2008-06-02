// -*- C++ -*-
//
// Package:    MCProcessFilter07
// Class:      MCProcessFilter07
// 
/**\class MCProcessFilter07 MCProcessFilter07.cc JetMETAnalysis/CSA07Skimming/src/MCProcessFilter07.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jochen Cammin
//         Created:  Mon Jan 14 21:54:50 CST 2008
//         Adapted from CSA06Skimming package
// $Id: MCProcessFilter07.cc,v 1.2 2008/01/23 21:22:16 cammin Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace std;

//
// class declaration
//

class MCProcessFilter07 : public edm::EDFilter {
   public:
      explicit MCProcessFilter07(const edm::ParameterSet&);
      ~MCProcessFilter07();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
  std::string label_;
  std::vector<int> processID;
  std::vector<double> pthatMin;
  std::vector<double> pthatMax;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MCProcessFilter07::MCProcessFilter07(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
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
  if (pthatMin.size() > 1 &&  processID.size() != pthatMin.size()
      || pthatMax.size() > 1 && processID.size() != pthatMax.size()) {
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


MCProcessFilter07::~MCProcessFilter07()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
MCProcessFilter07::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle< int > genProcessID;
   iEvent.getByLabel( "genEventProcID", genProcessID );
   int ThisprocessID = *genProcessID;
 
   Handle< double > genEventScale;
   iEvent.getByLabel( "genEventScale", genEventScale );
   double Thispthat = *genEventScale;

   bool accepted = false;

   // do the selection -- processID 0 is always accepted
   for (unsigned int i = 0; i < processID.size(); i++){
     if (processID[i] == ThisprocessID || processID[i] == 0) {

       if ( Thispthat > pthatMin[i] &&  Thispthat < pthatMax[i] ) {
         accepted = true;
       }

     }
   }

   if (accepted){ return true; } else {return false;}


}

// ------------ method called once each job just before starting event loop  ------------
void 
MCProcessFilter07::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MCProcessFilter07::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MCProcessFilter07);
