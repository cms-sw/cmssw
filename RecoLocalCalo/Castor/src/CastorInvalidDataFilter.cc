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
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ErrorSummaryEntry.h"

//
// class declaration
//

class CastorInvalidDataFilter : public edm::global::EDFilter<> {
public:
  explicit CastorInvalidDataFilter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::EDGetTokenT<std::vector<edm::ErrorSummaryEntry> > tok_summary_;
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
CastorInvalidDataFilter::CastorInvalidDataFilter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  tok_summary_ = consumes<std::vector<edm::ErrorSummaryEntry> >(edm::InputTag("logErrorHarvester"));
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool CastorInvalidDataFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  edm::Handle<std::vector<ErrorSummaryEntry> > summary;
  iEvent.getByToken(tok_summary_, summary);

  bool invalid = false;
  //std::cout << " logError summary size = " << summary->size() << std::endl;
  for (size_t i = 0; i < summary->size(); i++) {
    ErrorSummaryEntry error = (*summary)[i];
    //std::cout << " category = " << error.category << " module = " << error.module << " severity = "
    //        << error.severity.getName() << " count = " << error.count << std::endl;
    if (error.category == "Invalid Data" && error.module == "CastorRawToDigi:castorDigis")
      invalid = true;
  }

  return !invalid;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void CastorInvalidDataFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(CastorInvalidDataFilter);
