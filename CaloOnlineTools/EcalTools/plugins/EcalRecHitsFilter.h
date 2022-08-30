// -*- C++ -*-
//
// Package:   EcalRecHitsFilter
// Class:     EcalRecHitsFilter
//
//class EcalRecHitsFilter EcalRecHitsFilter.cc
//
// Original Author:
//         Created:  We May 14 10:10:52 CEST 2008
//

#ifndef EcalRecHitsFilter_H
#define EcalRecHitsFilter_H

// system include files
#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "TFile.h"
#include <string>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>

//
// class declaration
//

class EcalRecHitsFilter : public edm::one::EDFilter<> {
public:
  explicit EcalRecHitsFilter(const edm::ParameterSet&);
  ~EcalRecHitsFilter() override;

private:
  void beginJob() override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  const int NumBadXtalsThreshold_;
  const edm::EDGetTokenT<EcalRecHitCollection> EBRecHitCollection_;
  const double EnergyCut;

  TH1F* nRecHitsGreater1GevPerEvent_hist;
  TH2F* nRecHitsGreater1GevPerEvent_hist_MAP;
  TFile* file;
};

#endif
