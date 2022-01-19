#ifndef LHEGenericMassFilter_h
#define LHEGenericMassFilter_h
// -*- C++ -*-
//
// Package:    LHEGenericMassFilter
// Class:      LHEGenericMassFilter
// 
/* 

 Description: Filter to select events within a given mass range for an arbitrary number of given particle(s).

*/
// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

//
// class declaration
//

class LHEGenericMassFilter : public edm::global::EDFilter<> {
public:
  explicit LHEGenericMassFilter(const edm::ParameterSet&);
  ~LHEGenericMassFilter() override;

private:
  bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  void endJob() override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<LHEEventProduct> src_;
  const int numRequired_;              // number of particles required to pass filter
  const std::vector<int> particleID_;  // vector of particle IDs to look for
  const double minMass_;
  const double maxMass_;
};

#endif
