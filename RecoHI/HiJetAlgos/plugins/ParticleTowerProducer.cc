// -*- C++ -*-
//
// Package:    RecoHI/HiJetAlgos
// Class:      ParticleTowerProducer
//
/**\class ParticleTowerProducer RecoHI/HiJetAlgos/plugins/ParticleTowerProducer.cc
 Description: [one line class summary]
 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz,32 4-A08,+41227673039,
//         Created:  Thu Jan 20 19:53:58 CET 2011
//
//

#include "RecoHI/HiJetAlgos/plugins/ParticleTowerProducer.h"

#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "TMath.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ParticleTowerProducer::ParticleTowerProducer(const edm::ParameterSet& iConfig) : geo_(nullptr),
  src_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("src"))),
  useHF_(iConfig.getParameter<bool>("useHF"))
 {
  //register your products

  produces<CaloTowerCollection>();

  //now do what ever other initialization is needed
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void ParticleTowerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  geo_ = pG.product();

  resetTowers(iEvent, iSetup);

  auto const& inputs = iEvent.get(src_);
  for (auto const& particle : inputs) {
    double eta = particle.eta();

    int ieta = eta2ieta(eta);
    int iphi = phi2iphi(particle.phi(), ieta);

    if (!useHF_ && abs(ieta) > 29)
      continue;

    EtaPhi ep(ieta, iphi);
    towers_[ep] += particle.et();
  }

  auto prod = std::make_unique<CaloTowerCollection>();

  for (auto const& tower : towers_) {
    EtaPhi ep = tower.first;
    double et = tower.second;

    int ieta = ep.first;
    int iphi = ep.second;

    CaloTowerDetId newTowerId(ieta, iphi);  // totally dummy id

    if (et > 0) {
      if (!useHF_ && abs(ieta) > 29)
        continue;

      // currently sets et = pt, mass to zero
      // pt, eta, phi, mass
      reco::Particle::PolarLorentzVector p4(et, ieta2eta(ieta), iphi2phi(iphi, ieta), 0.);

      GlobalPoint point(p4.x(), p4.y(), p4.z());
      prod->emplace_back(newTowerId, et, 0, 0, 0, 0, p4, point, point);
    }
  }

  //For reference, Calo Tower Constructors

  /*
   CaloTower(const CaloTowerDetId& id,
             double emE, double hadE, double outerE,
             int ecal_tp, int hcal_tp,
             const PolarLorentzVector p4,
       GlobalPoint emPosition, GlobalPoint hadPosition);
   CaloTower(const CaloTowerDetId& id,
             double emE, double hadE, double outerE,
             int ecal_tp, int hcal_tp,
             const LorentzVector p4,
       GlobalPoint emPosition, GlobalPoint hadPosition);
   */

  iEvent.put(std::move(prod));
}

// ------------ method called once each job just before starting event loop  ------------
void ParticleTowerProducer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void ParticleTowerProducer::endJob() {}

void ParticleTowerProducer::resetTowers(edm::Event& iEvent, const edm::EventSetup& iSetup) { towers_.clear(); }

// Taken from FastSimulation/CalorimeterProperties/src/HCALProperties.cc
// Note this returns an abs(ieta)
int ParticleTowerProducer::eta2ieta(double eta) const {
  // binary search in the array of towers eta edges

  int ieta = 0;

  while (fabs(eta) > etaedge[ieta]) {
    ++ieta;
  }

  if (eta < 0)
    ieta = -ieta;
  return ieta;
}

int ParticleTowerProducer::phi2iphi(double phi, int ieta) const {
  if (phi < 0)
    phi += 2. * TMath::Pi();
  else if (phi > 2. * TMath::Pi())
    phi -= 2. * TMath::Pi();

  int Nphi = 72;
  int n = 1;
  if (abs(ieta) > 20)
    n = 2;
  if (abs(ieta) >= 40)
    n = 4;

  int iphi = (int)TMath::Ceil(phi / 2.0 / TMath::Pi() * Nphi / n);

  iphi = n * (iphi - 1) + 1;

  return iphi;
}

double ParticleTowerProducer::iphi2phi(int iphi, int ieta) const {
  double phi = 0;
  int Nphi = 72;

  int n = 1;
  if (abs(ieta) > 20)
    n = 2;
  if (abs(ieta) >= 40)
    n = 4;

  int myphi = (iphi - 1) / n + 1;

  phi = 2. * TMath::Pi() * (myphi - 0.5) / Nphi * n;
  while (phi > TMath::Pi())
    phi -= 2. * TMath::Pi();

  return phi;
}

double ParticleTowerProducer::ieta2eta(int ieta) const {
  int sign = 1;
  if (ieta < 0) {
    sign = -1;
    ieta = -ieta;
  }

  double eta = sign * (etaedge[ieta] + etaedge[ieta - 1]) / 2.;
  return eta;
}

// define this as a plug-in
DEFINE_FWK_MODULE(ParticleTowerProducer);
