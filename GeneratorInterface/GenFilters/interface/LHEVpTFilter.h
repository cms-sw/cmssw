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

// system include files
#include <memory>
#include <iostream>

// user include files
#include "Math/Vector4D.h"
#include "Math/Vector4Dfwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

//
// class declaration
//

class LHEVpTFilter : public edm::EDFilter {
public:
  explicit LHEVpTFilter(const edm::ParameterSet&);
  ~LHEVpTFilter() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<LHEEventProduct> src_;
  std::vector<lhef::HEPEUP::FiveVector> lheParticles;
  std::vector<ROOT::Math::PxPyPzEVector> lepCands;

  double vptMin_;    // number of particles required to pass filter
  double vptMax_;    // number of particles required to pass filter
  int totalEvents_;  // counters
  int passedEvents_;
};
#endif
