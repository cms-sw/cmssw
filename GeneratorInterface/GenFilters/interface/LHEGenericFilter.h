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
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

//
// class declaration
//

class LHEGenericFilter : public edm::EDFilter {
 public:
  explicit LHEGenericFilter(const edm::ParameterSet&);
  ~LHEGenericFilter();
  
 private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  
  // ----------member data ---------------------------
  
  edm::EDGetTokenT<LHEEventProduct> src_;
  int numRequired_;                // number of particles required to pass filter
  bool acceptMore_;                // if true (default), accept numRequired or more.
                                   // if false, accept events with exactly equal to numRequired.
  std::vector<int> particleID_;    // vector of particle IDs to look for
  int totalEvents_;                // counters
  int passedEvents_;
};
#endif
