// -*- C++ -*-
//
// Package:    TagProbeEdmFilter
// Class:      TagProbeEdmFilter
// 
/**\class TagProbeEdmFilter TagProbeEdmFilter.cc PhysicsTools/TagProbeEdmFilter/src/TagProbeEdmFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Nadia Adam
//         Created:  Fri Jun  6 09:10:41 CDT 2008
// $Id: TagProbeEDMFilter.cc,v 1.2 2008/07/30 13:38:25 srappocc Exp $
//
//

// system include files

// Include files
#include "PhysicsTools/TagAndProbe/interface/TagProbeEDMFilter.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TagProbeEDMFilter::TagProbeEDMFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


TagProbeEDMFilter::~TagProbeEDMFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
TagProbeEDMFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   // Pass the event if there are some TP pairs ...
   Handle< vector<float> > tp_mass;

   try{ iEvent.getByLabel("TPEdm","TPmass",tp_mass); }
   catch(...)
   {
      LogWarning("TagAndProbeFilter") << "Could not extract TP pairs ";
      return false;
   }

   int nrTP = tp_mass->size();

   if( nrTP > 0 ) return true;
   else           return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
TagProbeEDMFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TagProbeEDMFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TagProbeEDMFilter);
