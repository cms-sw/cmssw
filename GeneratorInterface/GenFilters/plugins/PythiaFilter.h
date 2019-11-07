#ifndef PYTHIAFILTER_h
#define PYTHIAFILTER_h
// -*- C++ -*-
//
// Package:    PythiaFilter
// Class:      PythiaFilter
//
/**\class PythiaFilter PythiaFilter.cc IOMC/PythiaFilter/src/PythiaFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Filip Moortgat
//         Created:  Mon Jan 23 14:57:54 CET 2006
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

//
// class decleration
//
namespace edm {
  class HepMCProduct;
}

class PythiaFilter : public edm::global::EDFilter<> {
public:
  explicit PythiaFilter(const edm::ParameterSet&);
  ~PythiaFilter() override;

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // ----------member data ---------------------------

  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const int particleID;
  const double minpcut;
  const double maxpcut;
  const double minptcut;
  const double maxptcut;
  const double minetacut;
  const double maxetacut;
  const double minrapcut;
  const double maxrapcut;
  const double minphicut;
  const double maxphicut;

  const int status;
  const int motherID;
  const int processID;

  const double betaBoost;
};
#endif
