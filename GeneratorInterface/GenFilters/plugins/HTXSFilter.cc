// -*- C++ -*-
//
// Package:    HTXSFilter
// Class:      HTXSFilter
//
/**\class HTXSFilter HTXSFilter.cc user/HTXSFilter/plugins/HTXSFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Janek Bechtel
//         Created:  Fri, 10 May 2019 14:30:15 GMT
//
//

// system include files
#include <memory>
#include "GeneratorInterface/GenFilters/plugins/HTXSFilter.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "SimDataFormats/HTXS/interface/HiggsTemplateCrossSections.h"

HTXSFilter::HTXSFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<HTXS::HiggsClassification>(edm::InputTag("rivetProducerHTXS", "HiggsClassification"))),
      htxs_flags(iConfig.getUntrackedParameter("htxs_flags", std::vector<int>())) {}

HTXSFilter::~HTXSFilter() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HTXSFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  Handle<HTXS::HiggsClassification> cat;
  iEvent.getByToken(token_, cat);
  if (htxs_flags.empty()) {
    edm::LogInfo("HTXSFilter") << "Selection of HTXS flags to filter is empty. Filtering will not be applied."
                               << std::endl;
    return true;
  }
  if (std::find(htxs_flags.begin(), htxs_flags.end(), cat->stage1_1_cat_pTjet30GeV) != htxs_flags.end()) {
    return true;
  } else {
    return false;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HTXSFilter);
