#ifndef LHEGenericFilter_h
#define LHEGenericFilter_h
// -*- C++ -*-
//
// Package:    LHEGenericFilter
// Class:      LHEGenericFilter
//
/* 

 Description: Filter to select events with an arbitrary number of given particle(s).

 Implementation: derived from MCSingleParticleFilter
     
*/
//
// Original Author:  Roberto Covarelli
//         Created:  Wed Feb 29 04:22:16 CST 2012
//
//

// system include files
#include <vector>

// user include files
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

class LHEGenericFilter : public edm::global::EDFilter<> {
public:
  explicit LHEGenericFilter(const edm::ParameterSet&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  const edm::EDGetTokenT<LHEEventProduct> src_;
  const int numRequired_;              // number of particles required to pass filter
  const std::vector<int> particleID_;  // vector of particle IDs to look for
  enum Logic { LT, GT, EQ, NE };
  Logic whichlogic;
};
#endif
