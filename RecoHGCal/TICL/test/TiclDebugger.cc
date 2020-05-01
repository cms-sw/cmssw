//
// Original Author:  Marco Rovere
//         Created:  Fri May  1 07:21:02 CEST 2020
//
//
//
// system include files
#include <memory>
#include <iostream>
#include <numeric>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"

#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"

//
// class declaration
//

class TiclDebugger : public edm::one::EDAnalyzer<> {
public:
  explicit TiclDebugger(const edm::ParameterSet&);
  ~TiclDebugger() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  edm::InputTag trackstersMerge_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>> trackstersMergeToken_;
};

TiclDebugger::TiclDebugger(const edm::ParameterSet& iConfig)
    : trackstersMerge_(iConfig.getParameter<edm::InputTag>("trackstersMerge")) {
  edm::ConsumesCollector&& iC = consumesCollector();
  trackstersMergeToken_ = iC.consumes<std::vector<ticl::Trackster>>(trackstersMerge_);
}

TiclDebugger::~TiclDebugger() {}

void TiclDebugger::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  static const char* particle_kind[] = {"gam", "e", "mu", "pi0", "h", "h0", "?", "!"};
  using namespace edm;
  using std::begin;
  using std::end;
  using std::iota;
  using std::sort;

  edm::Handle<std::vector<ticl::Trackster>> trackstersMergeH;

  iEvent.getByToken(trackstersMergeToken_, trackstersMergeH);
  auto const& tracksters = *trackstersMergeH.product();
  std::vector<int> sorted_tracksters_idx(tracksters.size());
  iota(begin(sorted_tracksters_idx), end(sorted_tracksters_idx), 0);
  sort(begin(sorted_tracksters_idx), end(sorted_tracksters_idx), [&tracksters](int i, int j) {
    return tracksters[i].raw_energy() > tracksters[j].raw_energy();
  });

  for (auto const & t : sorted_tracksters_idx) {
    auto const & trackster = tracksters[t];
    auto const & probs = trackster.id_probabilities();
    // Sort probs in descending order
    std::vector<int> sorted_probs_idx(probs.size());
    iota(begin(sorted_probs_idx), end(sorted_probs_idx), 0);
    sort(begin(sorted_probs_idx), end(sorted_probs_idx), [&probs](int i, int j) {
      return probs[i] > probs[j];
    });

    std::cout << "TrksIdx: " << t << "\n bary: " << trackster.barycenter()
      << " baryEta: " << trackster.barycenter().eta()
      << " baryPhi: " << trackster.barycenter().phi()
      << "\n raw_energy: " << trackster.raw_energy()
      << " raw_em_energy: " << trackster.raw_em_energy()
      << "\n raw_pt: " << trackster.raw_pt()
      << " raw_em_pt: " << trackster.raw_em_pt()
      << "\n seedIdx: " << trackster.seedIndex()
      << "\n Probs: ";
    for (auto p_idx : sorted_probs_idx) {
      std::cout << "(" << particle_kind[p_idx] << "):" << probs[p_idx] << " ";
    }
    std::cout << "\n time: " << trackster.time() << "+/-" << trackster.timeError()
      << std::endl;
  }
}

void TiclDebugger::beginJob() {}

void TiclDebugger::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TiclDebugger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("trackstersMerge", edm::InputTag("ticlTrackstersMerge"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  descriptions.add("ticlDebugger", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(TiclDebugger);
