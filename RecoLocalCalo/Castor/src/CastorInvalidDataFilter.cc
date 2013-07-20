// -*- C++ -*-
//
// Package:    CastorInvalidDataFilter
// Class:      CastorInvalidDataFilter
// 
/**\class CastorInvalidDataFilter CastorInvalidDataFilter.cc RecoLocalCalo/CastorInvalidDataFilter/src/CastorInvalidDataFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  local user
//         Created:  Thu Apr 21 11:36:52 CEST 2011
// $Id: CastorInvalidDataFilter.cc,v 1.2 2013/02/27 23:02:53 eulisse Exp $
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

#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"

//
// class declaration
//

class CastorInvalidDataFilter : public edm::EDFilter {
   public:
      explicit CastorInvalidDataFilter(const edm::ParameterSet&);
      ~CastorInvalidDataFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      
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
CastorInvalidDataFilter::CastorInvalidDataFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


CastorInvalidDataFilter::~CastorInvalidDataFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
CastorInvalidDataFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   edm::Handle<std::vector<ErrorSummaryEntry> > summary;
   iEvent.getByLabel("logErrorHarvester",summary);

   bool invalid = false;
   //std::cout << " logError summary size = " << summary->size() << std::endl;
   for (size_t i=0;i<summary->size();i++) {
        ErrorSummaryEntry error = (*summary)[i];
        //std::cout << " category = " << error.category << " module = " << error.module << " severity = "
        //        << error.severity.getName() << " count = " << error.count << std::endl;
	if (error.category == "Invalid Data" && error.module == "CastorRawToDigi:castorDigis") invalid = true;

   }

   return !invalid;
}






// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CastorInvalidDataFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(CastorInvalidDataFilter);
