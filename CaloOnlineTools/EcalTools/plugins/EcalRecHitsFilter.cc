// -*- C++ -*-
//
// Package:   EcalRecHitsFilter
// Class:     EcalRecHitsFilter
//
//class EcalHighEnCosmicFilter EcalHighEnCosmicFilter.cc
//
// Original Author:

#include <memory>
#include <vector>
#include <map>
#include <set>

// user include files
#include "CaloOnlineTools/EcalTools/plugins/EcalRecHitsFilter.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"

using namespace edm;
using namespace std;
using namespace reco;

//
EcalRecHitsFilter::EcalRecHitsFilter(const edm::ParameterSet& iConfig)
    : NumBadXtalsThreshold_(iConfig.getUntrackedParameter<int>("NumberXtalsThreshold")),
      EBRecHitCollection_(
          consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EcalRecHitCollectionEB"))),
      EnergyCut(iConfig.getUntrackedParameter<double>("energycut")) {}

EcalRecHitsFilter::~EcalRecHitsFilter() {}

bool EcalRecHitsFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //int ievt = iEvent.id().event();
  const Handle<EcalRecHitCollection>& EBhits = iEvent.getHandle(EBRecHitCollection_);

  bool accepted = true;
  int nRecHitsGreater1GevPerEvent = 0;

  for (EcalRecHitCollection::const_iterator hitItr = EBhits->begin(); hitItr != EBhits->end(); ++hitItr) {
    EcalRecHit hit = (*hitItr);
    EBDetId det = hit.id();

    float ampli = hit.energy();
    if (ampli > EnergyCut /*1GeV*/) {
      nRecHitsGreater1GevPerEvent++;
      nRecHitsGreater1GevPerEvent_hist_MAP->Fill(det.iphi(), det.ieta());
    }
  }
  nRecHitsGreater1GevPerEvent_hist->Fill(nRecHitsGreater1GevPerEvent);
  if (nRecHitsGreater1GevPerEvent > NumBadXtalsThreshold_)
    accepted = false;
  return accepted;
}

void EcalRecHitsFilter::beginJob() {
  nRecHitsGreater1GevPerEvent_hist =
      new TH1F("nRecHitsGreater1GevPerEvent_hist", "nRecHitsGreater1GevPerEvent_hist", 65000, 0., 65000.);
  nRecHitsGreater1GevPerEvent_hist_MAP = new TH2F(
      "nRecHitsGreater1GevPerEvent_hist_MAP", "nRecHitsGreater1GevPerEvent_hist_MAP", 360, 1., 361., 171, -85., 86.);
}

void EcalRecHitsFilter::endJob() {
  edm::LogVerbatim("EcalTools") << "------EcalRecHitsFilter EndJob------>>>>>>>>>>";
  file = new TFile("RecHitFilter.root", "RECREATE");
  file->cd();
  nRecHitsGreater1GevPerEvent_hist_MAP->Write();
  nRecHitsGreater1GevPerEvent_hist->Write();
  file->Close();
}
