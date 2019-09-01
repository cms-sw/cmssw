#ifndef MinimumBiasFilter_H
#define MinimumBiasFilter_H
//
// Original Author:  Filippo Ambroglini
//         Created:  Fri Sep 29 17:10:41 CEST 2006
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// class declaration
//

class MinimumBiasFilter : public edm::EDFilter {
public:
  MinimumBiasFilter(const edm::ParameterSet&);
  ~MinimumBiasFilter() override{};

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  // ----------member data ---------------------------
  float theEventFraction;
};
#endif
