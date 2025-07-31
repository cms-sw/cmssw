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
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/Math/interface/PtEtaPhiMass.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"

using RecHitRef = edm::Ref<CSCRecHit2DCollection>;
using RecHitRefVector = edm::RefVector<CSCRecHit2DCollection>;

class cscMDSshowerTableProducer : public edm::global::EDProducer<> {
public:
  cscMDSshowerTableProducer(const edm::ParameterSet& iConfig)
      : geometryToken_(esConsumes<CSCGeometry, MuonGeometryRecord>()),
        dtgeometryToken_(esConsumes<DTGeometry, MuonGeometryRecord>()),
        rpcgeometryToken_(esConsumes<RPCGeometry, MuonGeometryRecord>()),
        inputToken_(consumes<CSCRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("recHitLabel"))),
        dtSegmentToken_(consumes<DTRecSegment4DCollection>(iConfig.getParameter<edm::InputTag>("segmentLabel"))),
        rpchitToken_(consumes<RPCRecHitCollection>(iConfig.getParameter<edm::InputTag>("rpcLabel"))),
        rParam_(iConfig.getParameter<double>("rParam")),
        nRechitMin_(iConfig.getParameter<int>("nRechitMin")),
        nStationThres_(iConfig.getParameter<int>("nStationThres")),
        stripErr_(iConfig.getParameter<double>("stripErr")),
        wireError_(iConfig.getParameter<double>("wireError")),
        pruneCut_(iConfig.getParameter<double>("pruneCut")),
        name_(iConfig.getParameter<std::string>("name")) {
    produces<nanoaod::FlatTable>(name_ + "Rechits");
    produces<nanoaod::FlatTable>(name_);
  }

  ~cscMDSshowerTableProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("recHitLabel")->setComment("input cscRechit collection");
    desc.add<edm::InputTag>("segmentLabel")->setComment("input dt segment collection for veto");
    desc.add<edm::InputTag>("rpcLabel")->setComment("input rpcRechit collection for veto");
    desc.add<double>("rParam", 0.4);
    desc.add<int>("nRechitMin", 50);
    desc.add<int>("nStationThres", 10);
    desc.add<double>("stripErr", 7.0);
    desc.add<double>("wireError", 8.6);
    desc.add<double>("pruneCut", 9.0);
    desc.add<std::string>("name", "cscRechits")->setComment("name of the output collection");
    descriptions.add("cscMDSshowerTable", desc);
  }
  float getWeightedTime(RecHitRefVector rechits) const;

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> geometryToken_;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtgeometryToken_;
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcgeometryToken_;
  edm::EDGetTokenT<CSCRecHit2DCollection> inputToken_;
  edm::EDGetTokenT<DTRecSegment4DCollection> dtSegmentToken_;
  edm::EDGetTokenT<RPCRecHitCollection> rpchitToken_;
  const double rParam_;
  const int nRechitMin_;                          // min number of rechits
  const int nStationThres_;                       // min number of rechits to count towards nStation
  const double stripErr_, wireError_, pruneCut_;  //constants for CSC time
  const std::string name_;
};

//From: https://github.com/cms-sw/cmssw/blob/master/RecoMuon/MuonIdentification/src/CSCTimingExtractor.cc#L165-L234
float cscMDSshowerTableProducer::getWeightedTime(RecHitRefVector rechits) const {
  bool modified = false;
  double totalWeightTimeVtx = 0;
  float timeVtx = 0;

  do {
    modified = false;
    totalWeightTimeVtx = 0;
    timeVtx = 0;
    for (auto const& rechit : rechits) {
      timeVtx += rechit->wireTime() * 1. / (wireError_ * wireError_);
      timeVtx += rechit->tpeak() * 1. / (stripErr_ * stripErr_);
      totalWeightTimeVtx += 1. / (wireError_ * wireError_);
      totalWeightTimeVtx += 1. / (stripErr_ * stripErr_);
    }
    timeVtx /= totalWeightTimeVtx;

    // cut away outliers
    double diff_tvtx_strip, diff_tvtx_wire;
    double chimax = 0.0;
    int tmmax = 0;
    for (size_t i = 0; i < rechits.size(); ++i) {
      const auto& rechit = rechits[i];
      //diff_tvtx = (cscHits[i].time - timeVtx) * (cscHits[i].time - timeVtx) * cscHits[i].error;
      diff_tvtx_strip = (rechit->tpeak() - timeVtx) * (rechit->tpeak() - timeVtx) * 1. / (stripErr_ * stripErr_);
      diff_tvtx_wire = (rechit->wireTime() - timeVtx) * (rechit->wireTime() - timeVtx) * 1. / (wireError_ * wireError_);

      if ((diff_tvtx_strip > chimax) || (diff_tvtx_wire > chimax)) {
        tmmax = i;
        chimax = std::max(diff_tvtx_strip, diff_tvtx_wire);
      }
    }
    // cut away the outliers and repeat
    if (chimax > pruneCut_) {
      rechits.erase(rechits.begin() + tmmax);
      modified = true;
    }
  } while (modified);

  return timeVtx;
}

void cscMDSshowerTableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto const& geo = iSetup.getData(geometryToken_);
  auto const& dt_geo = iSetup.getData(dtgeometryToken_);
  auto const& rpc_geo = iSetup.getData(rpcgeometryToken_);
  auto const& rechits = iEvent.get(inputToken_);
  auto const& segments = iEvent.get(dtSegmentToken_);
  auto const& rpchits = iEvent.get(rpchitToken_);

  std::set<CSCDetId> unique_ids;
  std::vector<fastjet::PseudoJet> fjInput;

  edm::RefVector<CSCRecHit2DCollection> inputs;

  fastjet::JetDefinition jet_def(fastjet::cambridge_algorithm, rParam_);

  int recIt = 0;
  for (auto const& rechit : rechits) {
    LocalPoint recHitLocalPosition = rechit.localPosition();
    auto detid = rechit.cscDetId();
    auto thischamber = geo.chamber(detid);
    if (thischamber) {
      GlobalPoint globalPosition = thischamber->toGlobal(recHitLocalPosition);
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
  std::vector<float> cscRechitsX, cscRechitsY, cscRechitsZ, cscRechitsPhi, cscRechitsEta, cscRechitsE, cscRechitsTpeak,
      cscRechitsTwire;
  std::vector<int> cscRechitsNStrips, cscRechitsHitWire, cscRechitsWGroupsBX, cscRechitsNWireGroups, cscRechitsQuality,
      cscRechitsChamber, cscRechitsIChamber, cscRechitsStation;
  ;

  // MDS fields
  std::vector<float> clsX, clsY, clsZ, clsPhi, clsEta, clsTime, clsTimeSpread, clsTimeWeighted, clsTimeSpreadWeighted,
      clsAvgStation;
  std::vector<int> clsSize, clsNstation, cls_nME11, cls_nME12, clsUniqueChamber, cls_nMB1dtSeg, cls_nRE12hit,
      cls_nRB1hit;

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
    int nME12 = 0, nME11 = 0, nMB1dtSeg = 0, nRE12hit = 0, nRB1hit = 0;
    int nStation = 0;
    int totStation = 0;
    float avgStation = 0.0;
    float timeSpread = 0.0;
    float time = 0.0;
    float time_strip = 0.0;  // for timeSpread calculation
    double timeWeighted = 0.0;
    float timeSpreadWeighted = 0.0;

    std::map<int, int> station_count_map;

    //fill rechits fields
    for (auto const& rechit : rechits) {
      LocalPoint recHitLocalPosition = rechit->localPosition();
      auto detid = rechit->cscDetId();
      unique_ids.insert(detid.chamberId());
      auto thischamber = geo.chamber(detid);
      int endcap = CSCDetId::endcap(detid) == 1 ? 1 : -1;
      if (thischamber) {
        GlobalPoint globalPosition = thischamber->toGlobal(recHitLocalPosition);

        cscRechitsX.push_back(globalPosition.x());
        cscRechitsY.push_back(globalPosition.y());
        cscRechitsZ.push_back(globalPosition.z());
        cscRechitsPhi.push_back(globalPosition.phi());
        cscRechitsEta.push_back(globalPosition.eta());
        cscRechitsE.push_back(rechit->energyDepositedInLayer());  //not saved
        cscRechitsTpeak.push_back(rechit->tpeak());
        cscRechitsTwire.push_back(rechit->wireTime());
        cscRechitsQuality.push_back(rechit->quality());

        int stationRing = (CSCDetId::station(detid) * 10 + CSCDetId::ring(detid));
        if (CSCDetId::ring(detid) == 4)
          stationRing = (CSCDetId::station(detid) * 10 + 1);  // ME1/a has ring==4

        cscRechitsChamber.push_back(endcap * stationRing);
        cscRechitsIChamber.push_back(CSCDetId::chamber(detid));
        cscRechitsStation.push_back(endcap * CSCDetId::station(detid));
        cscRechitsNStrips.push_back(rechit->nStrips());
        cscRechitsHitWire.push_back(rechit->hitWire());
        cscRechitsWGroupsBX.push_back(rechit->wgroupsBX());
        cscRechitsNWireGroups.push_back(rechit->nWireGroups());

        //compute for cluster fields
        station_count_map[CSCDetId::station(detid)]++;
        if (stationRing == 11)
          nME11++;
        if (stationRing == 12)
          nME12++;
        time += (rechit->tpeak() + rechit->wireTime());
        time_strip += rechit->tpeak();
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
    float invN = 1.f / rechits.size();
    time = (time / 2.f) * invN;
    time_strip = time_strip * invN;

    // https://github.com/cms-sw/cmssw/blob/master/RecoMuon/MuonIdentification/src/CSCTimingExtractor.cc#L165-L234
    for (auto& rechit : rechits) {
      timeSpread += (rechit->tpeak() - time_strip) * (rechit->tpeak() - time_strip);
    }
    timeSpread = std::sqrt(timeSpread * invN);

    float i_clsEta = etaFromXYZ(fjJet.px() * invN, fjJet.py() * invN, fjJet.pz() * invN);
    float i_clsPhi = std::atan2(fjJet.py() * invN, fjJet.px() * invN);

    //MB1 DT seg
    for (auto const& segment : segments) {
      LocalPoint localPosition = segment.localPosition();
      auto geoid = segment.geographicalId();
      DTChamberId dtdetid = DTChamberId(geoid);
      auto thischamber = dt_geo.chamber(dtdetid);
      if (thischamber) {
        GlobalPoint globalPosition = thischamber->toGlobal(localPosition);
        float eta = globalPosition.eta();
        float phi = globalPosition.phi();
        if (dtdetid.station() == 1 && reco::deltaR(eta, phi, i_clsEta, i_clsPhi) < 0.4)
          nMB1dtSeg++;
      }
    }
    //RPC hits
    for (auto const& rechit : rpchits) {
      LocalPoint recHitLocalPosition = rechit.localPosition();
      auto geoid = rechit.geographicalId();
      RPCDetId rpcdetid = RPCDetId(geoid);
      auto thischamber = rpc_geo.chamber(rpcdetid);

      if (thischamber) {
        GlobalPoint globalPosition = thischamber->toGlobal(recHitLocalPosition);
        float eta = globalPosition.eta();
        float phi = globalPosition.phi();

        //RE12 hits
        if (rpcdetid.station() == 1 && rpcdetid.ring() == 2 && abs(rpcdetid.region()) == 1 &&
            reco::deltaR(eta, phi, i_clsEta, i_clsPhi) < 0.4)
          nRE12hit++;
        //RB1 hits
        if (rpcdetid.station() == 1 && rpcdetid.region() == 0 && reco::deltaR(eta, phi, i_clsEta, i_clsPhi) < 0.4)
          nRB1hit++;
      }
    }

    //fill cluster fields
    clsSize.push_back(rechits.size());
    // cluster position is the average position of the constituent rechits
    clsX.push_back(fjJet.px() * invN);
    clsY.push_back(fjJet.py() * invN);
    clsZ.push_back(fjJet.pz() * invN);
    clsEta.push_back(i_clsEta);
    clsPhi.push_back(i_clsPhi);

    clsTime.push_back(time);
    clsTimeSpread.push_back(timeSpread);
    cls_nME11.push_back(nME11);
    cls_nME12.push_back(nME12);
    clsNstation.push_back(nStation);
    clsAvgStation.push_back(avgStation);
    clsUniqueChamber.push_back(unique_ids.size());
    cls_nMB1dtSeg.push_back(nMB1dtSeg);
    cls_nRE12hit.push_back(nRE12hit);
    cls_nRB1hit.push_back(nRB1hit);

    //cluster time weighted with unc. and pruned outliers
    timeWeighted = cscMDSshowerTableProducer::getWeightedTime(rechits);

    for (auto& rechit : rechits) {
      timeSpreadWeighted += (timeWeighted - rechit->tpeak()) * (timeWeighted - rechit->tpeak());
    }
    timeSpreadWeighted = std::sqrt(timeSpreadWeighted * invN);

    clsTimeWeighted.push_back(timeWeighted);
    clsTimeSpreadWeighted.push_back(timeSpreadWeighted);
  }
  auto cscRechitTab = std::make_unique<nanoaod::FlatTable>(cscRechitsX.size(), name_ + "Rechits", false, false);

  cscRechitTab->addColumn<float>("X", cscRechitsX, "csc rechit X");
  cscRechitTab->addColumn<float>("Y", cscRechitsY, "csc rechit Y");
  cscRechitTab->addColumn<float>("Z", cscRechitsZ, "csc rechit Z");
  cscRechitTab->addColumn<float>("Phi", cscRechitsPhi, "csc rechit Phi");
  cscRechitTab->addColumn<float>("Eta", cscRechitsEta, "csc rechit Eta");
  cscRechitTab->addColumn<float>("E", cscRechitsE, "csc rechit Energy deposited in layer");
  cscRechitTab->addColumn<float>("Tpeak", cscRechitsTpeak, "csc rechit time from cathode");
  cscRechitTab->addColumn<float>("Twire", cscRechitsTwire, "csc rechit time from anode");
  cscRechitTab->addColumn<int>("Quality", cscRechitsQuality, "csc rechit quality");
  cscRechitTab->addColumn<int>("Chamber", cscRechitsChamber, "csc rechit station-Ring");
  cscRechitTab->addColumn<int>("IChamber", cscRechitsIChamber, "csc rechit chamber in ring");
  cscRechitTab->addColumn<int>("Station", cscRechitsStation, "csc rechit station");
  cscRechitTab->addColumn<int>("NStrips", cscRechitsNStrips, "csc rechit nstrips");
  cscRechitTab->addColumn<int>("WGroupsBX", cscRechitsWGroupsBX, "csc rechit wire group BX");
  cscRechitTab->addColumn<int>("HitWire", cscRechitsHitWire, "csc rechit hit wire");
  cscRechitTab->addColumn<int>("NWireGroups", cscRechitsNWireGroups, "csc rechit n wire groups");

  iEvent.put(std::move(cscRechitTab), name_ + "Rechits");

  auto clsTab = std::make_unique<nanoaod::FlatTable>(clsSize.size(), name_, false, false);

  clsTab->addColumn<int>("size", clsSize, "cluster Size");
  clsTab->addColumn<float>("x", clsX, "cluster X");
  clsTab->addColumn<float>("y", clsY, "cluster Y");
  clsTab->addColumn<float>("z", clsZ, "cluster Z");
  clsTab->addColumn<float>("phi", clsPhi, "cluster Phi");
  clsTab->addColumn<float>("eta", clsEta, "cluster Eta");
  clsTab->addColumn<float>("time", clsTime, "cluster Time");
  clsTab->addColumn<float>("timeSpread", clsTimeSpread, "cluster TimeSpread");
  clsTab->addColumn<float>("timeWeighted", clsTimeWeighted, "cluster TimeWeighted");
  clsTab->addColumn<float>("timeSpreadWeighted", clsTimeSpreadWeighted, "cluster TimeSpreadWeighted");
  clsTab->addColumn<int>("nStation", clsNstation, "cluster nStation");
  clsTab->addColumn<int>("uniqueChamber", clsUniqueChamber, "cluster unique chambers");
  clsTab->addColumn<float>("avgStation", clsAvgStation, "cluster AvgStation");
  clsTab->addColumn<int>("nME11", cls_nME11, "cluster nME11");
  clsTab->addColumn<int>("nME12", cls_nME12, "cluster nME12");
  clsTab->addColumn<int>("nMB1dtSeg", cls_nMB1dtSeg, "cluster nMB1dtSeg");
  clsTab->addColumn<int>("nRE12hit", cls_nRE12hit, "cluster nRE12hit");
  clsTab->addColumn<int>("nRB1hit", cls_nRB1hit, "cluster nRB1hit");

  iEvent.put(std::move(clsTab), name_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(cscMDSshowerTableProducer);
