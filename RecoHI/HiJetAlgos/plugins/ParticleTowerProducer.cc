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

#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoHI/HiJetAlgos/plugins/HITowerHelper.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <utility>

class ParticleTowerProducer : public edm::stream::EDProducer<> {
public:
  explicit ParticleTowerProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  int eta2ieta(double eta) const;
  int phi2iphi(double phi, int ieta) const;
  double ieta2eta(int ieta) const;
  double iphi2phi(int iphi, int ieta) const;
  // ----------member data ---------------------------

  edm::EDGetTokenT<reco::PFCandidateCollection> src_;
  const bool useHF_;

  // tower edges from fast sim, used starting at index 30 for the HF
  static constexpr int ietaMax = 42;
};
//
// constructors and destructor
//
ParticleTowerProducer::ParticleTowerProducer(const edm::ParameterSet& iConfig)
    : src_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      useHF_(iConfig.getParameter<bool>("useHF")) {
  produces<CaloTowerCollection>();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void ParticleTowerProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  typedef std::pair<int, int> EtaPhi;
  typedef std::map<EtaPhi, double> EtaPhiMap;
  EtaPhiMap towers_;
  towers_.clear();

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
  prod->reserve(towers_.size());
  for (auto const& tower : towers_) {
    EtaPhi ep = tower.first;
    double et = tower.second;

    int ieta = ep.first;
    int iphi = ep.second;

    CaloTowerDetId newTowerId(ieta, iphi);  // totally dummy id

    // currently sets et = pt, mass to zero
    // pt, eta, phi, mass
    reco::Particle::PolarLorentzVector p4(et, ieta2eta(ieta), iphi2phi(iphi, ieta), 0.);
    GlobalPoint point(p4.x(), p4.y(), p4.z());
    prod->emplace_back(newTowerId, p4.e(), 0, 0, 0, 0, p4, point, point);
  }

  iEvent.put(std::move(prod));
}

// Taken from FastSimulation/CalorimeterProperties/src/HCALProperties.cc
// Note this returns an abs(ieta)
int ParticleTowerProducer::eta2ieta(double eta) const {
  // binary search in the array of towers eta edges

  int ieta = 1;
  double xeta = fabs(eta);
  while (xeta > hi::etaedge[ieta] && ieta < ietaMax - 1) {
    ++ieta;
  }

  if (eta < 0)
    ieta = -ieta;
  return ieta;
}

int ParticleTowerProducer::phi2iphi(double phi, int ieta) const {
  phi = angle0to2pi::make0To2pi(phi);
  int nphi = 72;
  int n = 1;
  if (abs(ieta) > 20)
    n = 2;
  if (abs(ieta) >= 40)
    n = 4;

  int iphi = (int)std::ceil(phi / 2.0 / M_PI * nphi / n);

  iphi = n * (iphi - 1) + 1;

  return iphi;
}

double ParticleTowerProducer::iphi2phi(int iphi, int ieta) const {
  double phi = 0;
  int nphi = 72;

  int n = 1;
  if (abs(ieta) > 20)
    n = 2;
  if (abs(ieta) >= 40)
    n = 4;

  int myphi = (iphi - 1) / n + 1;

  phi = 2. * M_PI * (myphi - 0.5) / nphi * n;
  while (phi > M_PI)
    phi -= 2. * M_PI;

  return phi;
}

double ParticleTowerProducer::ieta2eta(int ieta) const {
  int sign = 1;
  if (ieta < 0) {
    sign = -1;
    ieta = -ieta;
  }

  double eta = sign * (hi::etaedge[ieta] + hi::etaedge[ieta - 1]) / 2.;
  return eta;
}

void ParticleTowerProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // particleTowerProducer
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("particleFlow"));
  desc.add<bool>("useHF", true);
  descriptions.add("particleTowerProducer", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(ParticleTowerProducer);
