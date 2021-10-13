#include <memory>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/MuonReco/interface/MuonRecHitCluster.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/PseudoJet.hh"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/MuonReco/interface/MuonRecHitCluster.h"

typedef std::vector<reco::MuonRecHitCluster> RecHitClusterCollection;

template <typename Trait>
class RechitClusterProducerT : public edm::global::EDProducer<> {
  typedef typename Trait::RecHitRef RechitRef;
  typedef typename Trait::RecHitRefVector RecHitRefVector;

public:
  explicit RechitClusterProducerT(const edm::ParameterSet&);
  ~RechitClusterProducerT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  const edm::ESGetToken<typename Trait::GeometryType, MuonGeometryRecord> GeometryToken_;
  edm::EDGetTokenT<typename Trait::InputType> InputToken_;
  typedef std::vector<reco::MuonRecHitCluster> RecHitClusterCollection;

  const double rParam_;      // distance paramter
  const int nRechitMin_;     // min number of rechits
  const int nStationThres_;  // min number of rechits to count towards nStation
};

template <typename Trait>
RechitClusterProducerT<Trait>::RechitClusterProducerT(const edm::ParameterSet& iConfig)
    : GeometryToken_(esConsumes<typename Trait::GeometryType, MuonGeometryRecord>()),
      InputToken_(consumes<typename Trait::InputType>(iConfig.getParameter<edm::InputTag>("recHitLabel"))),
      rParam_(iConfig.getParameter<double>("rParam")),
      nRechitMin_(iConfig.getParameter<int>("nRechitMin")),
      nStationThres_(iConfig.getParameter<int>("nStationThres")) {
  produces<RecHitClusterCollection>();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename Trait>
void RechitClusterProducerT<Trait>::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& iSetup) const {
  auto const& geo = iSetup.getData(GeometryToken_);
  auto const& rechits = ev.get(InputToken_);

  std::vector<fastjet::PseudoJet> fjInput;
  RecHitRefVector inputs_;

  fastjet::JetDefinition jet_def(fastjet::cambridge_algorithm, rParam_);

  inputs_.clear();
  fjInput.clear();

  Trait RecHitTrait;

  int recIt = 0;
  for (auto const& rechit : rechits) {
    LocalPoint RecHitLocalPosition = rechit.localPosition();
    auto detid = RecHitTrait.detid(rechit);
    auto thischamber = geo.chamber(detid);
    if (thischamber) {
      GlobalPoint globalPosition = thischamber->toGlobal(RecHitLocalPosition);
      float x = globalPosition.x();
      float y = globalPosition.y();
      float z = globalPosition.z();
      RechitRef ref = RechitRef(&rechits, recIt);
      inputs_.push_back(ref);
      fjInput.push_back(fastjet::PseudoJet(x, y, z, globalPosition.mag()));
      fjInput.back().set_user_index(recIt);
    }
    recIt++;
  }
  fastjet::ClusterSequence clus_seq(fjInput, jet_def);

  //keep all the clusters
  double ptmin = 0.0;
  std::vector<fastjet::PseudoJet> fjJets = fastjet::sorted_by_pt(clus_seq.inclusive_jets(ptmin));

  auto clusters = std::make_unique<RecHitClusterCollection>();
  if (!fjJets.empty()) {
    for (unsigned int ijet = 0; ijet < fjJets.size(); ++ijet) {
      // get the fastjet jet
      const fastjet::PseudoJet& fjJet = fjJets[ijet];

      // skip if the cluster has too few rechits
      if (int(fjJet.constituents().size()) < nRechitMin_)
        continue;
      // get the constituents from fastjet
      RecHitRefVector rechits;
      for (unsigned int i = 0; i < fjJet.constituents().size(); i++) {
        auto index = fjJet.constituents()[i].user_index();
        if (index >= 0 && static_cast<unsigned int>(index) < inputs_.size()) {
          rechits.push_back(inputs_[index]);
        }
      }

      //Derive cluster properties
      int nStation = 0;
      int totStation = 0;
      float avgStation = 0.0;
      std::map<int, int> station_count_map;
      for (auto& rechit : rechits) {
        station_count_map[RecHitTrait.station(*rechit)]++;
      }
      //station statistics
      std::map<int, int>::iterator it;
      for (auto const& [station, count] : station_count_map) {
        if (count >= nStationThres_) {
          nStation++;
          avgStation += (station) * (count);
          totStation += (count);
        }
      }
      if (totStation != 0) {
        avgStation = avgStation / totStation;
      }
      float invN = 1.f / rechits.size();
      // cluster position is the average position of the constituent rechits
      float jetX = fjJet.px() * invN;
      float jetY = fjJet.py() * invN;
      float jetZ = fjJet.pz() * invN;

      math::RhoEtaPhiVectorF position(
          std::sqrt(jetX * jetX + jetY * jetY), etaFromXYZ(jetX, jetY, jetZ), std::atan2(jetY, jetX));
      Trait::emplace_back(clusters.get(), position, nStation, avgStation, rechits);
    }
  }
  ev.put(std::move(clusters));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename Trait>
void RechitClusterProducerT<Trait>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("nRechitMin", 50);
  desc.add<double>("rParam", 0.4);
  desc.add<int>("nStationThres", 10);
  desc.add<edm::InputTag>("recHitLabel", edm::InputTag(Trait::recHitLabel()));
  descriptions.add(Trait::producerName(), desc);
}
struct DTRecHitTrait {
  using GeometryType = DTGeometry;
  using InputType = DTRecHitCollection;
  using RecHitRef = edm::Ref<DTRecHitCollection>;
  using RecHitRefVector = edm::RefVector<DTRecHitCollection>;
  static std::string recHitLabel() { return "dt1DRecHits"; }
  static std::string producerName() { return "DTrechitClusterProducer"; }

  static int station(const DTRecHit1DPair& dtRechit) { return detid(dtRechit).station(); }
  static DTChamberId detid(const DTRecHit1DPair& dtRechit) {
    DetId geoid = dtRechit.geographicalId();
    DTChamberId dtdetid = DTChamberId(geoid);
    return dtdetid;
  }
  static void emplace_back(RecHitClusterCollection* clusters,
                           math::RhoEtaPhiVectorF const& position,
                           int nStation,
                           float avgStation,
                           RecHitRefVector const& rechits) {
    // compute nMB1, nMB2
    int nMB1 = 0;
    int nMB2 = 0;
    for (auto& rechit : rechits) {
      DetId geoid = rechit->geographicalId();
      DTChamberId dtdetid = DTChamberId(geoid);
      if (dtdetid.station() == 1)
        nMB1++;
      if (dtdetid.station() == 2)
        nMB2++;
    }
    //set time, timespread, nME11,nME12 to 0
    reco::MuonRecHitCluster cls(position, rechits.size(), nStation, avgStation, 0.0, 0.0, 0, 0, 0, 0, nMB1, nMB2);
    clusters->emplace_back(cls);
  }
};

struct CSCRecHitTrait {
  using GeometryType = CSCGeometry;
  using InputType = CSCRecHit2DCollection;
  using RecHitRef = edm::Ref<CSCRecHit2DCollection>;
  using RecHitRefVector = edm::RefVector<CSCRecHit2DCollection>;
  static std::string recHitLabel() { return "csc2DRecHits"; }
  static std::string producerName() { return "CSCrechitClusterProducer"; }

  static int station(const CSCRecHit2D& cscRechit) { return CSCDetId::station(detid(cscRechit)); }
  static CSCDetId detid(const CSCRecHit2D& cscRechit) { return cscRechit.cscDetId(); }
  static void emplace_back(RecHitClusterCollection* clusters,
                           math::RhoEtaPhiVectorF const& position,
                           int nStation,
                           float avgStation,
                           RecHitRefVector const& rechits) {
    int nME11 = 0;
    int nME12 = 0;
    int nME41 = 0;
    int nME42 = 0;
    float timeSpread = 0.0;
    float time = 0.0;
    float time_strip = 0.0;  // for timeSpread calculation
    for (auto& rechit : rechits) {
      CSCDetId cscdetid = rechit->cscDetId();
      int stationRing = (CSCDetId::station(cscdetid) * 10 + CSCDetId::ring(cscdetid));
      if (CSCDetId::ring(cscdetid) == 4)
        stationRing = (CSCDetId::station(cscdetid) * 10 + 1);  // ME1/a has ring==4
      if (stationRing == 11)
        nME11++;
      if (stationRing == 12)
        nME12++;
      if (stationRing == 41)
        nME41++;
      if (stationRing == 42)
        nME42++;
      time += (rechit->tpeak() + rechit->wireTime());
      time_strip += rechit->tpeak();
    }
    float invN = 1.f / rechits.size();
    time = (time / 2.f) * invN;
    time_strip = time_strip * invN;

    //derive cluster statistics
    for (auto& rechit : rechits) {
      timeSpread += (rechit->tpeak() - time_strip) * (rechit->tpeak() - time_strip);
    }
    timeSpread = std::sqrt(timeSpread * invN);

    //set nMB1,nMB2 to 0
    reco::MuonRecHitCluster cls(
        position, rechits.size(), nStation, avgStation, time, timeSpread, nME11, nME12, nME41, nME42, 0, 0);
    clusters->emplace_back(cls);
  }
};

using DTrechitClusterProducer = RechitClusterProducerT<DTRecHitTrait>;
using CSCrechitClusterProducer = RechitClusterProducerT<CSCRecHitTrait>;
DEFINE_FWK_MODULE(DTrechitClusterProducer);
DEFINE_FWK_MODULE(CSCrechitClusterProducer);
