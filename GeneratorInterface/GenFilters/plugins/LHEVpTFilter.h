#ifndef LHEVpTFilter_h
#define LHEVpTFilter_h
// -*- C++ -*-
//
// Package:    LHEVpTFilter
// Class:      LHEVpTFilter
//
/* 

 Description: Filter to select events with V pT in a given range.
 (Based on LHEGenericFilter)

     
*/
//

// user include files
#include "Math/Vector4D.h"
#include "Math/Vector4Dfwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

//
// class declaration
//

class LHEVpTFilter : public edm::global::EDFilter<> {
public:
  explicit LHEVpTFilter(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<LHEEventProduct> src_;
  const double vptMin_;  // number of particles required to pass filter
  const double vptMax_;  // number of particles required to pass filter
};
#endif
