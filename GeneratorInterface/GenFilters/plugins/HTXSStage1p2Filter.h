#ifndef HTXSSTAGE1P2_FILTER_h
#define HTXSSTAGE1P2_FILTER_h
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
//         Created:  Mon, 03 August 2024 09:28:47 GMT
//
//

// system include files
#include <memory>

// user include files
#include "SimDataFormats/HTXS/interface/HiggsTemplateCrossSections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//
namespace edm {
  class HiggsClassification;
}

class HTXSStage1p2Filter : public edm::global::EDFilter<> {
public:
  explicit HTXSStage1p2Filter(const edm::ParameterSet&);
  ~HTXSStage1p2Filter() override;

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------

  const edm::EDGetTokenT<HTXS::HiggsClassification> token_;
  const std::vector<int> htxs_flags;
};
#endif
