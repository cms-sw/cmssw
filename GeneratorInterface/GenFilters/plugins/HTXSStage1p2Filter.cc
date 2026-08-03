// -*- C++ -*-
//
// Package:    HTXSStage1p2Filter
// Class:      HTXSStage1p2Filter
//
/**\class HTXSStage1p2Filter HTXSStage1p2Filter.cc user/HTXSStage1p2Filter/plugins/HTXSStage1p2Filter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Tom Runting
//         Created:  Mon, 03 August 2024 09:29:16 GMT
//
//

// system include files
#include <memory>
#include "GeneratorInterface/GenFilters/plugins/HTXSStage1p2Filter.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "SimDataFormats/HTXS/interface/HiggsTemplateCrossSections.h"

HTXSStage1p2Filter::HTXSStage1p2Filter(const edm::ParameterSet& iConfig)
    : token_(consumes<HTXS::HiggsClassification>(edm::InputTag("rivetProducerHTXS", "HiggsClassification"))),
      htxs_flags(iConfig.getUntrackedParameter("htxs_flags", std::vector<int>())) {}

HTXSStage1p2Filter::~HTXSStage1p2Filter() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HTXSStage1p2Filter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  Handle<HTXS::HiggsClassification> cat;
  iEvent.getByToken(token_, cat);
  if (htxs_flags.empty()) {
    edm::LogInfo("HTXSStage1p2Filter") << "Selection of HTXS flags to filter is empty. Filtering will not be applied."
                                       << std::endl;
    return true;
  }
  if (std::find(htxs_flags.begin(), htxs_flags.end(), cat->stage1_2_cat_pTjet30GeV) != htxs_flags.end()) {
    return true;
  } else {
    return false;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HTXSStage1p2Filter);
