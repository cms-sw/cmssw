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
#include <set>

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

class LHEPtFilter : public edm::EDFilter {
 public:
  explicit LHEPtFilter(const edm::ParameterSet&);
  ~LHEPtFilter();
  
 private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  
  // ----------member data ---------------------------
  
  edm::EDGetTokenT<LHEEventProduct> src_;
  std::vector<lhef::HEPEUP::FiveVector> lheParticles;
  std::vector<ROOT::Math::PxPyPzEVector> cands;
  std::vector<int> pdgIdVec_;
  std::set<int> pdgIds_;        // Set of PDG Ids to include
  double ptMin_;                // number of particles required to pass filter
  double ptMax_;                // number of particles required to pass filter
  int totalEvents_;                // counters
  int passedEvents_;
};
#endif
