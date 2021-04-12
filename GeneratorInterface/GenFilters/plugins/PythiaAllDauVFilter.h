#ifndef PYTHIAALLDAUVFILTER_h
#define PYTHIAALLDAUVFILTER_h
// -*- C++ -*-
//
// Package:    PythiaAllDauVFilter
// Class:      PythiaAllDauVFilter
//
/**\class PythiaAllDauVFilter PythiaAllDauVFilter.cc 

 Description: Filter events using MotherId and ChildrenIds infos
   		Accepts if event has a specified Mother with only specified daughters and all of the daughters  complies to respective pT and eta Cuts

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Daniele Pedrini, PythiaDauVFilter
//                   Aravind T S    ,   Updated and Modified to PythiaAllDauVFilter
//         Created:  Apr 29 2008
//                   Apr 12 2021
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

class PythiaAllDauVFilter : public edm::global::EDFilter<> {
public:
  explicit PythiaAllDauVFilter(const edm::ParameterSet&);
  ~PythiaAllDauVFilter() override;

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const int fVerbose;
  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int> dauIDs, antiDauIDs;
  const int particleID;
  int antiParticleID;
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
