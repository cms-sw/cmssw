// -*- C++ -*-
//
// Package:   BeamSplash
// Class:     BeamSplash
//
// Original Author:  Luca Malgeri

#ifndef BeamSplash_H
#define BeamSplash_H

// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


//
// class declaration
//


class BeamSplash : public edm::EDFilter {
public:
  explicit BeamSplash( const edm::ParameterSet & );
  ~BeamSplash();
  
private:
  virtual bool filter ( edm::Event &, const edm::EventSetup&) override;
  
  edm::InputTag EBRecHitCollection_;
  edm::InputTag EERecHitCollection_;
  edm::InputTag HBHERecHitCollection_;
  double EnergyCutTot;
  double EnergyCutEcal;
  double EnergyCutHcal;
  bool applyfilter;

  
};

#endif
