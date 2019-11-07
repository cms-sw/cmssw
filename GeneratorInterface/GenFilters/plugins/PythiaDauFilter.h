#ifndef PYTHIADAUFILTER_h
#define PYTHIADAUFILTER_h
// -*- C++ -*-
//
// Package:    PythiaDauFilter
// Class:      PythiaDauFilter
//
/**\class PythiaDauFilter PythiaDauFilter.cc 

 Description: Filter events using MotherId and ChildrenIds infos

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Daniele Pedrini
//         Created:  Apr 29 2008
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

class PythiaDauFilter : public edm::global::EDFilter<> {
public:
  explicit PythiaDauFilter(const edm::ParameterSet&);
  ~PythiaDauFilter() override;

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------memeber function----------------------

  // ----------member data ---------------------------

  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int> dauIDs;
  const int particleID;
  const bool chargeconju;
  const int ndaughters;
  const double minptcut;
  const double maxptcut;
  const double minetacut;
  const double maxetacut;
  std::unique_ptr<Pythia8::Pythia> fLookupGen;  // this instance is for accessing particleData information
};
#endif
