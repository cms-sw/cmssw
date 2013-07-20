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
// $Id: TagProbeMassEDMFilter.cc,v 1.3 2010/10/31 17:40:03 wmtan Exp $
//
//

// system include files

// Include files
#include "DPGAnalysis/Skims/interface/TagProbeMassEDMFilter.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TagProbeMassEDMFilter::TagProbeMassEDMFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
  tpMapName = iConfig.getParameter<std::string>("tpMapName");

}


TagProbeMassEDMFilter::~TagProbeMassEDMFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
TagProbeMassEDMFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   // Pass the event if there are some TP pairs ...
   Handle< std::vector<float> > tp_mass;

   iEvent.getByLabel(tpMapName,"TPmass",tp_mass);
   if (!tp_mass.isValid())
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
TagProbeMassEDMFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TagProbeMassEDMFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TagProbeMassEDMFilter);
