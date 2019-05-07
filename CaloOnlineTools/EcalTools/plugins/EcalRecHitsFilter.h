// -*- C++ -*-
//
// Package:   EcalRecHitsFilter
// Class:     EcalRecHitsFilter
//
// class EcalRecHitsFilter EcalRecHitsFilter.cc
//
// Original Author:
//         Created:  We May 14 10:10:52 CEST 2008
//

#ifndef EcalRecHitsFilter_H
#define EcalRecHitsFilter_H

// system include files
#include <map>
#include <memory>
#include <set>
#include <vector>

// user include files
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "TFile.h"
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <string>

//
// class declaration
//

class EcalRecHitsFilter : public edm::EDFilter {
public:
  explicit EcalRecHitsFilter(const edm::ParameterSet &);
  ~EcalRecHitsFilter() override;

private:
  void beginJob() override;
  bool filter(edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  double EnergyCut;
  int NumBadXtalsThreshold_;
  edm::InputTag EBRecHitCollection_;

  TH1F *nRecHitsGreater1GevPerEvent_hist;
  TH2F *nRecHitsGreater1GevPerEvent_hist_MAP;
  TFile *file;
};

#endif
