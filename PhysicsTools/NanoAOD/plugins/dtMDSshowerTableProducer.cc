#include <memory>
#include <set>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"

using RecHitRef = edm::Ref<DTRecHitCollection>;
using RecHitRefVector = edm::RefVector<DTRecHitCollection>;

class dtMDSshowerTableProducer : public edm::global::EDProducer<> {
public:
  dtMDSshowerTableProducer(const edm::ParameterSet& iConfig)
      : dtgeometryToken_(esConsumes<DTGeometry, MuonGeometryRecord>()),
        rpcgeometryToken_(esConsumes<RPCGeometry, MuonGeometryRecord>()),
        inputToken_(consumes<DTRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitLabel"))),
        rpchitToken_(consumes<RPCRecHitCollection>(iConfig.getParameter<edm::InputTag>("rpcLabel"))),
        rParam_(iConfig.getParameter<double>("rParam")),
        nRechitMin_(iConfig.getParameter<int>("nRechitMin")),
        nStationThres_(iConfig.getParameter<int>("nStationThres")),
        name_(iConfig.getParameter<std::string>("name")) {
    produces<nanoaod::FlatTable>(name_ + "Rechits");
    produces<nanoaod::FlatTable>(name_);
  }

  ~dtMDSshowerTableProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("recHitLabel")->setComment("input dtRechit collection");
    desc.add<edm::InputTag>("rpcLabel")->setComment("input rpcRechit collection for veto");
    desc.add<double>("rParam", 0.4);
    desc.add<int>("nRechitMin", 50);
    desc.add<int>("nStationThres", 10);
    desc.add<std::string>("name", "dt1DRecHits")->setComment("name of the output collection");
    descriptions.add("dtMDSshowerTable", desc);
  }
  float getWeightedTime(RecHitRefVector rechits) const;

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtgeometryToken_;
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcgeometryToken_;
  edm::EDGetTokenT<DTRecHitCollection> inputToken_;
  edm::EDGetTokenT<RPCRecHitCollection> rpchitToken_;
  const double rParam_;
  const int nRechitMin_;     // min number of rechits
  const int nStationThres_;  // min number of rechits to count towards nStation
  const std::string name_;
};

void dtMDSshowerTableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto const& dt_geo = iSetup.getData(dtgeometryToken_);
  auto const& rpc_geo = iSetup.getData(rpcgeometryToken_);
  auto const& rechits = iEvent.get(inputToken_);
  auto const& rpchits = iEvent.get(rpchitToken_);

  std::set<DTChamberId> unique_ids;
  std::vector<fastjet::PseudoJet> fjInput;

  RecHitRefVector inputs;

  fastjet::JetDefinition jet_def(fastjet::cambridge_algorithm, rParam_);

  int recIt = 0;
  for (auto const& rechit : rechits) {
    LocalPoint recHitLocalPosition = rechit.localPosition();
    auto geoid = rechit.geographicalId();
    DTChamberId dtdetid = DTChamberId(geoid);
    DTLayerId dtlayerid = DTLayerId(geoid);

    auto thischamber = dt_geo.chamber(dtdetid);
    auto thislayer = dt_geo.layer(dtlayerid);

    if (thischamber) {
      GlobalPoint globalPosition;
      if (thislayer) {
        globalPosition = thislayer->toGlobal(recHitLocalPosition);
      } else {
        globalPosition = thischamber->toGlobal(recHitLocalPosition);
      }
      float x = globalPosition.x();
      float y = globalPosition.y();
      float z = globalPosition.z();
      RecHitRef ref = RecHitRef(&rechits, recIt);
      inputs.push_back(ref);
      fjInput.push_back(fastjet::PseudoJet(x, y, z, globalPosition.mag()));
      fjInput.back().set_user_index(recIt);
    }
    recIt++;
  }
  fastjet::ClusterSequence clus_seq(fjInput, jet_def);

  //keep all the clusters
  double ptmin = 0.0;
  std::vector<fastjet::PseudoJet> fjJets = clus_seq.inclusive_jets(ptmin);

  // Constituent rechit fields
  std::vector<float> dtRechitsX, dtRechitsY, dtRechitsZ, dtRechitsPhi, dtRechitsEta;
  std::vector<int> dtRechitsLayer, dtRechitsSuperLayer, dtRechitsSector, dtRechitsStation, dtRechitsWheel;

  // MDS fields
  std::vector<float> clsX, clsY, clsZ, clsPhi, clsEta, clsAvgStation;
  std::vector<int> clsSize, clsNstation, clsWheel, clsUniqueChamber, cls_nRB1hit, cls_nRPC, clsBX;

  for (auto const& fjJet : fjJets) {
    // skip if the cluster has too few rechits
    if (int(fjJet.constituents().size()) < nRechitMin_)
      continue;
    // get the constituents from fastjet
    RecHitRefVector rechits;
    for (auto const& constituent : fjJet.constituents()) {
      auto index = constituent.user_index();
      if (index >= 0 && static_cast<unsigned int>(index) < inputs.size()) {
        rechits.push_back(inputs[index]);
      }
    }

    //Derive cluster properties
    int size_z = 0, size_xy = 0, size = 0;
    float avg_x_sl2(0.0), avg_y_sl2(0.0), avg_z_sl2(0.0);
    float avg_x(0.0), avg_y(0.0), avg_z(0.0);
    int nStation = 0;
    int totStation = 0;
    float avgStation = 0.0;

    std::map<int, int> station_count_map;

    //fill rechits fields
    for (auto const& rechit : rechits) {
      LocalPoint recHitLocalPosition = rechit->localPosition();
      auto geoid = rechit->geographicalId();

      DTChamberId dtdetid = DTChamberId(geoid);
      DTLayerId dtlayerid = DTLayerId(geoid);

      unique_ids.insert(dtdetid);

      auto thischamber = dt_geo.chamber(dtdetid);
      auto thislayer = dt_geo.layer(dtlayerid);

      if (thischamber) {
        if (thislayer) {
          GlobalPoint globalPosition = thislayer->toGlobal(recHitLocalPosition);
          dtRechitsX.push_back(globalPosition.x());
          dtRechitsY.push_back(globalPosition.y());
          dtRechitsZ.push_back(globalPosition.z());
          dtRechitsPhi.push_back(globalPosition.phi());
          dtRechitsEta.push_back(globalPosition.eta());
          dtRechitsLayer.push_back(dtlayerid.layer());
          dtRechitsSuperLayer.push_back(dtlayerid.superlayer());
          // use xy-coordinates from SL1/SL3 when available
          if (dtlayerid.superlayer() == 1 || dtlayerid.superlayer() == 3) {
            avg_x += globalPosition.x();
            avg_y += globalPosition.y();
            avg_z += globalPosition.z();
            size_xy++;
          }
          // use z-coordinates from SL2 when available
          else if (dtlayerid.superlayer() == 2) {
            avg_x_sl2 += globalPosition.x();
            avg_y_sl2 += globalPosition.y();
            avg_z_sl2 += globalPosition.z();
            size_z++;
          }
        } else {
          GlobalPoint globalPosition = thischamber->toGlobal(recHitLocalPosition);
          dtRechitsX.push_back(globalPosition.x());
          dtRechitsY.push_back(globalPosition.y());
          dtRechitsZ.push_back(globalPosition.z());
          dtRechitsPhi.push_back(globalPosition.phi());
          dtRechitsEta.push_back(globalPosition.eta());
          dtRechitsLayer.push_back(0);       //default value;
          dtRechitsSuperLayer.push_back(0);  //default value
          avg_x += globalPosition.x();
          avg_y += globalPosition.y();
          avg_z += globalPosition.z();
        }
        dtRechitsSector.push_back(dtdetid.sector());
        dtRechitsStation.push_back(dtdetid.station());
        dtRechitsWheel.push_back(dtdetid.wheel());
        size++;

        //compute for cluster fields
        station_count_map[dtdetid.station()]++;
      }
    }
    //station statistics
    std::map<int, int>::iterator it;
    for (auto const& [station, count] : station_count_map) {
      if (count >= nStationThres_) {
        nStation++;
        avgStation += station * count;
        totStation += count;
      }
    }
    if (totStation != 0) {
      avgStation = avgStation / totStation;
    }

    //for DT correct position, calculate average Z using sl2 and average XY using sl1/3
    if (size_xy > 0 && size_z > 0) {  // both SL1/SL3 and SL2 rechits
      avg_x = avg_x / size_xy;
      avg_y = avg_y / size_xy;
      avg_z = avg_z_sl2 / size_z;
    } else if (size_xy == 0 && size_z > 0)  // only SL2 rechits
    {
      avg_x = avg_x_sl2 / size_z;
      avg_y = avg_y_sl2 / size_z;
      avg_z = avg_z_sl2 / size_z;
    } else if (size_xy > 0 && size_z == 0)  // no SL2 rechits
    {
      avg_x = avg_x / size_xy;
      avg_y = avg_y / size_xy;
      avg_z = avg_z / size_xy;
    } else  // no SL information
    {
      avg_x = avg_x / size;
      avg_y = avg_y / size;
      avg_z = avg_z / size;
    }

    float i_clsEta = etaFromXYZ(avg_x, avg_y, avg_z);
    float i_clsPhi = std::atan2(avg_y, avg_x);
    int i_clsWheel = (std::abs(avg_z) < 126.8)            ? 0
                     : (avg_z > 126.8 && avg_z < 395.4)   ? 1
                     : (avg_z < -126.8 && avg_z > -395.4) ? -1
                     : (avg_z < 0)                        ? -2
                                                          : 2;

    // RPC hits
    std::map<int, int> bxCounts;  // bx of matched RPC hits : counts
    int nRB1hit = 0;
    for (auto const& rechit : rpchits) {
      LocalPoint recHitLocalPosition = rechit.localPosition();
      auto geoid = rechit.geographicalId();
      RPCDetId rpcdetid = RPCDetId(geoid);
      auto thischamber = rpc_geo.chamber(rpcdetid);

      // Only matching RB hits
      if (rpcdetid.region() != 0)
        continue;
      if (thischamber) {
        GlobalPoint globalPosition = thischamber->toGlobal(recHitLocalPosition);

        //match to RPC hits with dPhi<0.5 and same wheel in DT
        if (reco::deltaPhi(globalPosition.phi(), i_clsPhi) < 0.5 && rpcdetid.ring() == i_clsWheel) {
          //RB1 hits
          if (rpcdetid.station() == 1)
            nRB1hit++;
          bxCounts[rechit.BunchX()]++;
        }
      }
    }
    // find the mode of BX
    int modeBX = 0, maxCount = 0, i_cls_nRPC = 0;
    for (const auto& [bx, count] : bxCounts) {
      i_cls_nRPC += count;
      if (count > maxCount) {
        modeBX = bx;
        maxCount = count;
      }
    }

    //fill cluster fields
    clsSize.push_back(rechits.size());
    // cluster position is the average position of the constituent rechits
    clsX.push_back(avg_x);
    clsY.push_back(avg_y);
    clsZ.push_back(avg_z);
    clsEta.push_back(i_clsEta);
    clsPhi.push_back(i_clsPhi);
    clsBX.push_back(modeBX);
    clsNstation.push_back(nStation);
    clsAvgStation.push_back(avgStation);
    clsWheel.push_back(i_clsWheel);
    clsUniqueChamber.push_back(unique_ids.size());
    cls_nRB1hit.push_back(nRB1hit);
    cls_nRPC.push_back(i_cls_nRPC);
  }
  auto dtRechitTab = std::make_unique<nanoaod::FlatTable>(dtRechitsX.size(), name_ + "Rechits", false, false);

  dtRechitTab->addColumn<float>("X", dtRechitsX, "dt rechit X");
  dtRechitTab->addColumn<float>("Y", dtRechitsY, "dt rechit Y");
  dtRechitTab->addColumn<float>("Z", dtRechitsZ, "dt rechit Z");
  dtRechitTab->addColumn<float>("Phi", dtRechitsPhi, "dt rechit Phi");
  dtRechitTab->addColumn<float>("Eta", dtRechitsEta, "dt rechit Eta");
  dtRechitTab->addColumn<int>("Layer", dtRechitsLayer, "dt rechit Layer");
  dtRechitTab->addColumn<int>("SuperLayer", dtRechitsSuperLayer, "dt rechit SuperLayer");
  dtRechitTab->addColumn<int>("Sector", dtRechitsSector, "dt rechit sector");
  dtRechitTab->addColumn<int>("Station", dtRechitsStation, "dt rechit station");
  dtRechitTab->addColumn<int>("Wheel", dtRechitsWheel, "dt rechit nstrips");

  iEvent.put(std::move(dtRechitTab), name_ + "Rechits");

  auto clsTab = std::make_unique<nanoaod::FlatTable>(clsSize.size(), name_, false, false);

  clsTab->addColumn<int>("size", clsSize, "cluster Size");
  clsTab->addColumn<float>("x", clsX, "cluster X");
  clsTab->addColumn<float>("y", clsY, "cluster Y");
  clsTab->addColumn<float>("z", clsZ, "cluster Z");
  clsTab->addColumn<float>("phi", clsPhi, "cluster Phi");
  clsTab->addColumn<float>("eta", clsEta, "cluster Eta");
  clsTab->addColumn<int>("bx", clsBX, "cluster BX");
  clsTab->addColumn<int>("wheel", clsWheel, "cluster wheel");
  clsTab->addColumn<int>("nStation", clsNstation, "cluster nStation");
  clsTab->addColumn<int>("uniqueChamber", clsUniqueChamber, "cluster unique chambers");
  clsTab->addColumn<float>("avgStation", clsAvgStation, "cluster AvgStation");
  clsTab->addColumn<int>("nRPC", cls_nRPC, "cluster nRPC");
  clsTab->addColumn<int>("nRB1hit", cls_nRB1hit, "cluster nRB1hit");

  iEvent.put(std::move(clsTab), name_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(dtMDSshowerTableProducer);
