// -*- C++ -*-
//
// Package:   SelectHFMinBias
// Class:     SelectHFMinBias
//
// Original Author:  Luca Malgeri

#ifndef SelectHFMinBias_H
#define SelectHFMinBias_H

// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class SelectHFMinBias : public edm::stream::EDFilter<> {
public:
  explicit SelectHFMinBias(const edm::ParameterSet &);
  ~SelectHFMinBias() override;

private:
  bool filter(edm::Event &, const edm::EventSetup &) override;
};

#endif
