#ifndef PYTHIADAUVFILTER_h
#define PYTHIADAUVFILTER_h
// -*- C++ -*-
//
// Package:    PythiaDauVFilter
// Class:      PythiaDauVFilter
//
/**\class PythiaDauVFilter PythiaDauVFilter.cc 

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

class PythiaDauVFilter : public edm::global::EDFilter<> {
public:
  explicit PythiaDauVFilter(const edm::ParameterSet&);
  ~PythiaDauVFilter() override;

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const int fVerbose;
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int> dauIDs;
  const int particleID;
  const int motherID;
  const bool chargeconju;
  const int ndaughters;
  std::vector<double> minptcut;
  const double maxptcut;
  std::vector<double> minetacut;
  std::vector<double> maxetacut;
  std::unique_ptr<Pythia8::Pythia> fLookupGen;  // this instance is for accessing particleData information
};
#endif
