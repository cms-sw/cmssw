#ifndef MCMultiParticleFilter_h
#define MCMultiParticleFilter_h
// -*- C++ -*-
//
// Package:    MCMultiParticleFilter
// Class:      MCMultiParticleFilter
// 
/* 

 Description: Filter to select events with an arbitrary number of given particle(s).

 Implementation: derived from MCSingleParticleFilter
     
*/
//
// Original Author:  Paul Lujan
//         Created:  Wed Feb 29 04:22:16 CST 2012
//
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

class MCMultiParticleFilter : public edm::EDFilter {
 public:
  explicit MCMultiParticleFilter(const edm::ParameterSet&);
  ~MCMultiParticleFilter();
  
 private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  
  // ----------member data ---------------------------
  
  edm::InputTag src_;              // input tag
  int numRequired_;                // number of particles required to pass filter
  bool acceptMore_;                // if true (default), accept numRequired or more.
                                   // if false, accept events with exactly equal to numRequired.
  std::vector<int> particleID_;    // vector of particle IDs to look for
  // the three next variables can either be a vector of length 1 (in which case the same
  // value is used for all particle IDs) or of length equal to the length of ParticleID (in which
  // case the corresponding value is used for each).
  std::vector<double> ptMin_;      // minimum Pt of particles
  std::vector<double> etaMax_;     // maximum fabs(eta) of particles
  std::vector<int> status_;        // status of particles
  int totalEvents_;                // counters
  int passedEvents_;
};
#endif
