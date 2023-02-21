#ifndef HTXS_FILTER_h
#define HTXS_FILTER_h
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

class HTXSFilter : public edm::global::EDFilter<> {
public:
  explicit HTXSFilter(const edm::ParameterSet&);
  ~HTXSFilter() override;

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------

  const edm::EDGetTokenT<HTXS::HiggsClassification> token_;
  const std::vector<int> htxs_flags;
};
#endif
