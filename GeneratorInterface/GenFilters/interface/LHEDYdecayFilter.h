#ifndef LHEZDECAYFILTER_h
#define LHEZDECAYFILTER_h
// -*- C++ -*-
//
// Package:    LHEDYdecayFilter
// Class:      LHEDYdecayFilter
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class LHEDYdecayFilter : public edm::EDFilter {
public:
  explicit LHEDYdecayFilter(const edm::ParameterSet&);
  ~LHEDYdecayFilter() override;

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------

  edm::InputTag src_;
  int leptonID_;
  bool verbose_;
};
#endif
