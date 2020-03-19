#ifndef PYTHIAPROBEFILTER_h
#define PYTHIAPROBEFILTER_h
// -*- C++ -*-
//
// Package:    PythiaProbeFilter
// Class:      PythiaProbeFilter
//
/**\class PythiaProbeFilter PythiaProbeFilter.cc 

 Description: Filter to exclude selected particles from passing pT,eta cuts etc. Usefull when we are interested in a decay that its daughters should not pass any cuts, but another particle of the same flavour should e.g B+B- production with B+->K+mumu forcing (probe side) and we want mu to come from B- (tag side)

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Georgios Karathanasis
//         Created:  Mar 14 2019
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Pythia8/Pythia.h"

//
// class decleration
//
namespace edm {
  class HepMCProduct;
}

class PythiaProbeFilter : public edm::global::EDFilter<> {
public:
  explicit PythiaProbeFilter(const edm::ParameterSet&);
  ~PythiaProbeFilter() override;

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  bool AlreadyExcludedCheck(std::vector<unsigned int> excludedList, unsigned int current_part) const;

private:
  // ----------memeber function----------------------

  // ----------member data ---------------------------

  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int> exclsisIDs;
  std::vector<int> exclauntIDs;
  const int particleID;
  const int MomID;
  const int GrandMomID;
  const bool chargeconju;
  const int nsisters;
  const int naunts;
  const double minptcut;
  const double maxptcut;
  const double minetacut;
  const double maxetacut;
  const bool countQEDCorPhotons;
  bool identicalParticle;
  std::unique_ptr<Pythia8::Pythia> fLookupGen;  // this instance is for accessing particleData information
};
#endif
