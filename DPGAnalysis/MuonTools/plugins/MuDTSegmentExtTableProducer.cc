/** \class MuDTSegmentExtTableProducer MuDTSegmentExtTableProducer.ccDPGAnalysis/MuonTools/src/MuDTSegmentExtTableProducer.cc
 *  
 * Helper class : the segment TableProducer for Phase-1 / Phase2 segments (the DataFormat is the same)
 *
 * \author C. Battilana (INFN BO)
 *
*
*/

#include "DPGAnalysis/MuonTools/interface/MuBaseFlatTableProducer.h"

#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

#include <vector>
#include <array>

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

class MuDTSegmentExtTableProducer : public MuBaseFlatTableProducer {
public:
  /// Constructor
  MuDTSegmentExtTableProducer(const edm::ParameterSet&);

  /// Fill descriptors
  static void fillDescriptions(edm::ConfigurationDescriptions&);

protected:
  /// Fill tree branches for a given event
  void fillTable(edm::Event&) final;

  /// Get info from the ES by run
  void getFromES(const edm::Run&, const edm::EventSetup&) final;

  /// Get info from the ES for a given event
  void getFromES(const edm::EventSetup&) final;

private:
  static const int FIRST_LAYER{1};
  static const int LAST_LAYER{4};
  static const int THETA_SL{2};
  /// The segment token
  nano_mu::EDTokenHandle<DTRecSegment4DCollection> m_token;

  /// Fill rec-hit table
  bool m_fillHits;

  /// Fill segment extrapolation  table
  bool m_fillExtr;

  /// DT Geometry
  nano_mu::ESTokenHandle<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun> m_dtGeometry;

  /// Tracking Geometry
  nano_mu::ESTokenHandle<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> m_trackingGeometry;

  /// Handle DT trigger time pedestals
  std::unique_ptr<DTTTrigBaseSync> m_dtSync;
};

MuDTSegmentExtTableProducer::MuDTSegmentExtTableProducer(const edm::ParameterSet& config)
    : MuBaseFlatTableProducer{config},
      m_token{config, consumesCollector(), "src"},
      m_fillHits{config.getParameter<bool>("fillHits")},
      m_fillExtr{config.getParameter<bool>("fillExtrapolation")},
      m_dtGeometry{consumesCollector()},
      m_trackingGeometry{consumesCollector()} {
  produces<nanoaod::FlatTable>();
  if (m_fillHits) {
    produces<nanoaod::FlatTable>("hits");
  }
  if (m_fillExtr) {
    produces<nanoaod::FlatTable>("extr");
  }

  m_dtSync = DTTTrigSyncFactory::get()->create(config.getParameter<std::string>("tTrigMode"),
                                               config.getParameter<edm::ParameterSet>("tTrigModeConfig"),
                                               consumesCollector());
}

void MuDTSegmentExtTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("name", "dtSegment");
  desc.add<edm::InputTag>("src", edm::InputTag{"dt4DSegments"});
  desc.add<bool>("fillExtrapolation", true);
  desc.add<bool>("fillHits", true);

  desc.add<std::string>("tTrigMode", "DTTTrigSyncFromDB");

  edm::ParameterSetDescription tTrigModeParams;
  tTrigModeParams.add<bool>("doTOFCorrection", true);
  tTrigModeParams.add<int>("tofCorrType", 2);

  tTrigModeParams.add<double>("vPropWire", 24.4);
  tTrigModeParams.add<bool>("doWirePropCorrection", true);
  tTrigModeParams.add<int>("wirePropCorrType", 0);

  tTrigModeParams.add<std::string>("tTrigLabel", "");
  tTrigModeParams.add<bool>("doT0Correction", true);
  tTrigModeParams.add<std::string>("t0Label", "");
  tTrigModeParams.addUntracked<bool>("debug", false);

  desc.add<edm::ParameterSetDescription>("tTrigModeConfig", tTrigModeParams);

  descriptions.addWithDefaultLabel(desc);
}

void MuDTSegmentExtTableProducer::getFromES(const edm::Run& run, const edm::EventSetup& environment) {
  m_dtGeometry.getFromES(environment);
}

void MuDTSegmentExtTableProducer::getFromES(const edm::EventSetup& environment) {
  m_trackingGeometry.getFromES(environment);
  m_dtSync->setES(environment);
}

void MuDTSegmentExtTableProducer::fillTable(edm::Event& ev) {
  unsigned int nSegments{0};

  std::vector<float> seg4D_posLoc_x_SL1;
  std::vector<float> seg4D_posLoc_x_SL3;
  std::vector<float> seg4D_posLoc_x_midPlane;

  std::vector<uint32_t> seg4D_extr_begin;
  std::vector<uint32_t> seg4D_extr_end;

  std::vector<uint32_t> seg2D_hits_begin;
  std::vector<uint32_t> seg2D_hits_end;

  // segment extrapolation to DT layers filled, if m_fillExtr == true
  unsigned int nExtr{0};

  std::vector<float> seg4D_hitsExpPos;
  std::vector<float> seg4D_hitsExpPosCh;
  std::vector<int8_t> seg4D_hitsExpWire;

  // rec-hits vectors, filled if m_fillHits == true
  unsigned int nHits{0};

  std::vector<float> seg2D_hits_pos;
  std::vector<float> seg2D_hits_posCh;
  std::vector<float> seg2D_hits_posErr;
  std::vector<int8_t> seg2D_hits_side;
  std::vector<int8_t> seg2D_hits_wire;
  std::vector<int8_t> seg2D_hits_wirePos;
  std::vector<int8_t> seg2D_hits_layer;
  std::vector<int8_t> seg2D_hits_superLayer;
  std::vector<float> seg2D_hits_time;
  std::vector<float> seg2D_hits_timeCali;

  auto fillHits = [&](const DTRecSegment2D* seg, const GeomDet* chamb) {
    const auto& recHits = seg->specificRecHits();

    for (const auto& recHit : recHits) {
      ++nHits;

      const auto wireId = recHit.wireId();
      const auto layerId = wireId.layerId();
      const auto* layer = m_dtGeometry->layer(layerId);

      seg2D_hits_pos.push_back(recHit.localPosition().x());
      seg2D_hits_posCh.push_back(chamb->toLocal(layer->toGlobal(recHit.localPosition())).x());
      seg2D_hits_posErr.push_back(recHit.localPositionError().xx());

      seg2D_hits_side.push_back(recHit.lrSide());
      seg2D_hits_wire.push_back(wireId.wire());
      seg2D_hits_wirePos.push_back(layer->specificTopology().wirePosition(wireId.wire()));
      seg2D_hits_layer.push_back(layerId.layer());
      seg2D_hits_superLayer.push_back(layerId.superlayer());

      auto digiTime = recHit.digiTime();

      auto tTrig = m_dtSync->offset(wireId);

      seg2D_hits_time.push_back(digiTime);
      seg2D_hits_timeCali.push_back(digiTime - tTrig);
    }
  };

  auto segments4D = m_token.conditionalGet(ev);

  if (segments4D.isValid()) {
    auto chambIt = segments4D->id_begin();
    const auto chambEnd = segments4D->id_end();

    for (; chambIt != chambEnd; ++chambIt) {
      const auto& range = segments4D->get(*chambIt);

      for (auto segment4D = range.first; segment4D != range.second; ++segment4D) {
        auto station = (*chambIt).station();
        auto wheel = (*chambIt).wheel();
        auto sector = (*chambIt).sector();

        bool hasPhi = segment4D->hasPhi();
        bool hasZed = segment4D->hasZed();

        auto pos = segment4D->localPosition();
        auto dir = segment4D->localDirection();

        std::array<float, 2> xPosLocSL{{DEFAULT_DOUBLE_VAL, DEFAULT_DOUBLE_VAL}};
        std::array<bool, 2> hasPptSL{{false, false}};
        auto xPosLocMidPlane = DEFAULT_DOUBLE_VAL;

        const auto* chamb = m_dtGeometry->chamber(*chambIt);

        StraightLinePlaneCrossing segmentPlaneCrossing{
            chamb->toGlobal(pos).basicVector(), chamb->toGlobal(dir).basicVector(), anyDirection};

        if (hasPhi) {
          for (int iSL = 0; iSL < 2; ++iSL) {
            const auto* sl = chamb->superLayer(1 + iSL * 2);

            auto pptSL = segmentPlaneCrossing.position(sl->surface());
            hasPptSL[iSL] = pptSL.first;

            if (hasPptSL[iSL]) {
              GlobalPoint segExrapolationToSL(pptSL.second);
              xPosLocSL[iSL] = chamb->toLocal(segExrapolationToSL).x();
            }
          }
        }

        seg4D_posLoc_x_SL1.push_back(xPosLocSL[0]);
        seg4D_posLoc_x_SL3.push_back(xPosLocSL[1]);

        if (hasPptSL[0] && hasPptSL[1])
          xPosLocMidPlane = (xPosLocSL[0] + xPosLocSL[1]) * 0.5;

        seg4D_posLoc_x_midPlane.push_back(xPosLocMidPlane);

        const auto begin = seg4D_hitsExpPos.size();

        const auto size{station == 4 ? 8 : 12};

        nExtr += size;
        seg4D_extr_begin.push_back(begin);
        seg4D_extr_end.push_back(begin + size);

        const auto iSLs = station < 4 ? std::vector{1, 2, 3} : std::vector{1, 3};

        for (int iL = FIRST_LAYER; iL <= LAST_LAYER; ++iL) {
          for (const auto iSL : iSLs) {
            auto* layer = m_dtGeometry->layer(DTWireId{wheel, station, sector, iSL, iL, 2});
            auto ppt = segmentPlaneCrossing.position(layer->surface());

            bool success{ppt.first};  // check for failure

            auto expPos{DEFAULT_DOUBLE_VAL};
            auto expPosCh{DEFAULT_DOUBLE_VAL};
            auto expWire{DEFAULT_INT_VAL_POS};

            if (success) {
              GlobalPoint segExrapolationToLayer(ppt.second);
              LocalPoint segPosAtZWireLayer = layer->toLocal(segExrapolationToLayer);
              LocalPoint segPosAtZWireChamber = chamb->toLocal(segExrapolationToLayer);

              if (hasPhi && iSL != THETA_SL) {
                expPos = segPosAtZWireLayer.x();
                expPosCh = segPosAtZWireChamber.x();
                expWire = layer->specificTopology().channel(segPosAtZWireLayer);
              } else if (hasZed && iSL == THETA_SL) {
                expPos = segPosAtZWireLayer.x();
                expPosCh = segPosAtZWireChamber.y();
                expWire = layer->specificTopology().channel(segPosAtZWireLayer);
              }
            }

            seg4D_hitsExpPos.push_back(expPos);
            seg4D_hitsExpPosCh.push_back(expPosCh);
            seg4D_hitsExpWire.push_back(expWire);
          }
        }

        seg2D_hits_begin.push_back(seg2D_hits_pos.size());

        const GeomDet* geomDet = m_trackingGeometry->idToDet(segment4D->geographicalId());
        if (hasPhi) {
          fillHits(segment4D->phiSegment(), geomDet);
        }

        if (hasZed) {
          fillHits(segment4D->zSegment(), geomDet);
        }

        seg2D_hits_end.push_back(seg2D_hits_pos.size());
        ++nSegments;
      }
    }
  }

  auto table = std::make_unique<nanoaod::FlatTable>(nSegments, m_name, false, true);

  table->setDoc("DT segment information");

  addColumn(table, "seg4D_posLoc_x_SL1", seg4D_posLoc_x_SL1, "position x at SL1 in local coordinates - cm");
  addColumn(table, "seg4D_posLoc_x_SL3", seg4D_posLoc_x_SL3, "position x at SL3 in local coordinates - cm");
  addColumn(table,
            "seg4D_posLoc_x_midPlane",
            seg4D_posLoc_x_midPlane,
            "position x at SL1 - SL3 mid plane in local coordinates - cm");

  addColumn(table, "seg2D_hits_begin", seg2D_hits_begin, "begin() of range of quantities in the *_hits_* vectors");
  addColumn(table, "seg2D_hits_end", seg2D_hits_end, "end() of range of quantities in the *_hits_* vectors");

  addColumn(table, "seg4D_extr_begin", seg4D_extr_begin, "begin() of range of quantities in the *_extr_* vectors");
  addColumn(table, "seg4D_extr_end", seg4D_extr_end, "end() of range of quantities in the *_extr_* vectors");

  ev.put(std::move(table));

  if (m_fillHits) {
    auto tabHits = std::make_unique<nanoaod::FlatTable>(nHits, m_name + "_hits", false, false);

    tabHits->setDoc("Size of DT segment *_hits_* vectors");

    addColumn(tabHits, "pos", seg2D_hits_pos, "local x position of a hit in layer local coordinates");
    addColumn(tabHits, "posCh", seg2D_hits_posCh, "local x position of a hit in chamber local coordinates");
    addColumn(tabHits,
              "posErr",
              seg2D_hits_posErr,
              "local position error of a hit in layer local coordinates - xx component of error matrix");
    addColumn(tabHits, "side", seg2D_hits_side, "is hit on L/R side of a cell wire - 1/2 is R/L");
    addColumn(tabHits, "wire", seg2D_hits_wire, "hit wire number - range depends on chamber size");
    addColumn(tabHits, "wirePos", seg2D_hits_wirePos, "hit wire x position in layer local coordinates");
    addColumn(tabHits, "layer", seg2D_hits_layer, "hit layer number - range [1:4]");
    addColumn(tabHits,
              "superLayer",
              seg2D_hits_superLayer,
              "hit superlayer - [1:3] range"
              "<br />SL 1 and 3 are phi SLs"
              "<br />SL 2 is theta SL");
    addColumn(tabHits, "time", seg2D_hits_time, "digi time - ns, pedestal not subtracted");
    addColumn(tabHits, "timeCali", seg2D_hits_timeCali, "digi time - ns, pedestal subtracted");

    ev.put(std::move(tabHits), "hits");
  }

  if (m_fillExtr) {
    auto tabExtr = std::make_unique<nanoaod::FlatTable>(nExtr, m_name + "_extr", false, false);

    tabExtr->setDoc("Size of DT segment *_extr_* vectors");
    addColumn(tabExtr,
              "ExpPos",
              seg4D_hitsExpPos,
              "expected position of segment extrapolated"
              "<br />to a given layer in layer local coordinates - cm");

    addColumn(tabExtr,
              "ExpPosCh",
              seg4D_hitsExpPosCh,
              "expected position of segment extrapolated"
              "<br />to a given layer in chhamber local coordinates - cm");

    addColumn(tabExtr,
              "ExpWire",
              seg4D_hitsExpWire,
              "expected wire crossed by segment extrapolated"
              "<br />to a given layer - range depends on chamber size");

    ev.put(std::move(tabExtr), "extr");
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuDTSegmentExtTableProducer);
