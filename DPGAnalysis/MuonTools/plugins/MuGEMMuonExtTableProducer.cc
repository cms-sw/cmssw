#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"

#include <vector>

#include "DPGAnalysis/MuonTools/interface/MuBaseFlatTableProducer.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

class MuGEMMuonExtTableProducer : public MuBaseFlatTableProducer {
public:
  /// Constructor
  MuGEMMuonExtTableProducer(const edm::ParameterSet&);

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
  /// The RECO mu token
  nano_mu::EDTokenHandle<reco::MuonCollection> m_token;

  /// Fill matches table
  bool m_fillPropagated;

  /// GEM Geometry
  nano_mu::ESTokenHandle<GEMGeometry, MuonGeometryRecord, edm::Transition::BeginRun> m_gemGeometry;

  /// Transient Track Builder
  nano_mu::ESTokenHandle<TransientTrackBuilder, TransientTrackRecord> m_transientTrackBuilder;

  /// Muon service proxy
  std::unique_ptr<MuonServiceProxy> m_muonSP;
};

MuGEMMuonExtTableProducer::MuGEMMuonExtTableProducer(const edm::ParameterSet& config)
    : MuBaseFlatTableProducer{config},
      m_token{config, consumesCollector(), "src"},
      m_fillPropagated{config.getParameter<bool>("fillPropagated")},
      m_gemGeometry{consumesCollector()},
      m_transientTrackBuilder{consumesCollector(), "TransientTrackBuilder"},
      m_muonSP{std::make_unique<MuonServiceProxy>(config.getParameter<edm::ParameterSet>("ServiceParameters"),
                                                  consumesCollector())} {
  produces<nanoaod::FlatTable>();

  if (m_fillPropagated) {
    produces<nanoaod::FlatTable>("propagated");
  }
}

void MuGEMMuonExtTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("name", "muon");
  desc.add<edm::InputTag>("src", edm::InputTag{"muons"});

  desc.add<bool>("fillPropagated", true);
  desc.setAllowAnything();

  descriptions.addWithDefaultLabel(desc);
}

void MuGEMMuonExtTableProducer::getFromES(const edm::Run& run, const edm::EventSetup& environment) {
  m_gemGeometry.getFromES(environment);
}

void MuGEMMuonExtTableProducer::getFromES(const edm::EventSetup& environment) {
  m_transientTrackBuilder.getFromES(environment);
  m_muonSP->update(environment);
}

void MuGEMMuonExtTableProducer::fillTable(edm::Event& ev) {
  unsigned int nMuons{0};

  std::vector<bool> isCSC;
  std::vector<bool> isME11;

  std::vector<float> innermost_x;
  std::vector<float> innermost_y;
  std::vector<float> innermost_z;

  std::vector<float> outermost_x;
  std::vector<float> outermost_y;
  std::vector<float> outermost_z;

  unsigned int nProp{0};

  std::vector<uint32_t> propagated_muIdx;

  std::vector<bool> propagated_isincoming;
  std::vector<bool> propagated_isinsideout;
  std::vector<int8_t> propagated_region;
  std::vector<int8_t> propagated_layer;
  std::vector<int8_t> propagated_chamber;
  std::vector<int8_t> propagated_etaP;

  std::vector<float> propagatedLoc_x;
  std::vector<float> propagatedLoc_y;
  std::vector<float> propagatedLoc_z;
  std::vector<float> propagatedLoc_r;
  std::vector<float> propagatedLoc_phi;
  std::vector<float> propagatedLoc_dirX;
  std::vector<float> propagatedLoc_dirY;
  std::vector<float> propagatedLoc_dirZ;
  std::vector<float> propagatedLoc_errX;
  std::vector<float> propagatedLoc_errY;

  std::vector<float> propagatedGlb_x;
  std::vector<float> propagatedGlb_y;
  std::vector<float> propagatedGlb_z;
  std::vector<float> propagatedGlb_r;
  std::vector<float> propagatedGlb_phi;
  std::vector<float> propagatedGlb_errX;
  std::vector<float> propagatedGlb_errY;
  std::vector<float> propagatedGlb_phierr;
  std::vector<float> propagatedGlb_rerr;

  std::vector<float> propagated_EtaPartition_centerX;
  std::vector<float> propagated_EtaPartition_centerY;
  std::vector<float> propagated_EtaPartition_phiMax;
  std::vector<float> propagated_EtaPartition_phiMin;
  std::vector<float> propagated_EtaPartition_rMax;
  std::vector<float> propagated_EtaPartition_rMin;

  std::vector<int8_t> propagated_nME1hits;
  std::vector<int8_t> propagated_nME2hits;
  std::vector<int8_t> propagated_nME3hits;
  std::vector<int8_t> propagated_nME4hits;

  auto muons = m_token.conditionalGet(ev);

  // edm::ESHandle<Propagator>
  auto&& propagator_any = m_muonSP->propagator("SteppingHelixPropagatorAny");
  auto&& propagator_along = m_muonSP->propagator("SteppingHelixPropagatorAlong");
  auto&& propagator_opposite = m_muonSP->propagator("SteppingHelixPropagatorOpposite");

  if (!propagator_any.isValid() || !propagator_along.isValid() || !propagator_opposite.isValid()) {
    return;
  }

  if (muons.isValid() && m_transientTrackBuilder.isValid()) {
    //loop on recoMuons
    for (const auto& muon : (*muons)) {
      ++nMuons;

      bool is_csc = false;
      bool is_me11 = false;

      if (!muon.outerTrack().isNull()) {
        const auto track = muon.outerTrack().get();
        const auto outerTrackRef = muon.outerTrack();

        float p2_in = track->innerMomentum().mag2();
        float p2_out = track->outerMomentum().mag2();
        float pos_out = track->outerPosition().mag2();
        float pos_in = track->innerPosition().mag2();

        bool is_insideout = pos_in > pos_out;

        if (is_insideout) {
          std::swap(pos_in, pos_out);
          std::swap(p2_in, p2_out);
        }

        bool is_incoming = p2_out > p2_in;

        const auto&& transient_track = m_transientTrackBuilder->build(track);
        const auto& htp = transient_track.hitPattern();

        if (transient_track.isValid()) {
          const auto innerPosGlb{transient_track.innermostMeasurementState().globalPosition()};
          const auto outerPosGlb{transient_track.outermostMeasurementState().globalPosition()};

          innermost_x.push_back(innerPosGlb.x());
          innermost_y.push_back(innerPosGlb.y());
          innermost_z.push_back(innerPosGlb.z());
          outermost_x.push_back(outerPosGlb.x());
          outermost_y.push_back(outerPosGlb.y());
          outermost_z.push_back(outerPosGlb.z());
        } else {
          innermost_x.push_back(DEFAULT_DOUBLE_VAL);
          innermost_y.push_back(DEFAULT_DOUBLE_VAL);
          innermost_z.push_back(DEFAULT_DOUBLE_VAL);
          outermost_x.push_back(DEFAULT_DOUBLE_VAL);
          outermost_y.push_back(DEFAULT_DOUBLE_VAL);
          outermost_z.push_back(DEFAULT_DOUBLE_VAL);
          continue;
        }

        const auto&& start_state =
            is_insideout ? transient_track.outermostMeasurementState() : transient_track.innermostMeasurementState();
        auto& propagator = is_incoming ? propagator_along : propagator_opposite;

        auto recHitMu = outerTrackRef->recHitsBegin();
        auto recHitMuEnd = outerTrackRef->recHitsEnd();

        //loop on recHits which form the outerTrack
        for (; recHitMu != recHitMuEnd; ++recHitMu) {
          DetId detId{(*recHitMu)->geographicalId()};

          if (detId.det() == DetId::Muon && detId.subdetId() == MuonSubdetId::CSC) {
            is_csc = true;

            const CSCDetId csc_id{detId};
            // ME11 chambers consist of 2 subchambers: ME11a, ME11b.
            // In CMSSW they are referred as Stat. 1 Ring 1, Stat. 1 Ring. 4 respectively
            if (csc_id.station() == 1 && ((csc_id.ring() == 1) || (csc_id.ring() == 4)))
              is_me11 = true;
          }
        }  //loop on recHits

        //if at least one CSC hit is found, perform propagation
        if (is_csc) {
          // CSC Hits
          int8_t nME1_hits = 0;
          int8_t nME2_hits = 0;
          int8_t nME3_hits = 0;
          int8_t nME4_hits = 0;

          int nHits{htp.numberOfAllHits(htp.TRACK_HITS)};

          for (int i = 0; i != nHits; ++i) {
            uint32_t hit = htp.getHitPattern(htp.TRACK_HITS, i);
            int substructure = htp.getSubStructure(hit);
            int hittype = htp.getHitType(hit);

            if (substructure == 2 && hittype == 0) {  // CSC Hits
              int CSC_station = htp.getMuonStation(hit);

              switch (CSC_station) {
                case 1:
                  ++nME1_hits;
                  break;
                case 2:
                  ++nME2_hits;
                  break;
                case 3:
                  ++nME3_hits;
                  break;
                case 4:
                  ++nME4_hits;
                  break;
                default:
                  // do nothing
                  break;
              }
            }
          }
          //loop on GEM etaPartitions
          for (const auto& eta_partition : m_gemGeometry->etaPartitions()) {
            if (eta_partition->id().station() != 1) {
              continue;  //Only takes GE1/1
            }
            const GEMDetId&& gem_id = eta_partition->id();

            bool is_opposite_region = muon.eta() * gem_id.region() < 0;
            if (is_incoming xor is_opposite_region) {
              continue;  //Check on muon direction
            }
            const BoundPlane& bound_plane = eta_partition->surface();

            const auto& dest_state = propagator->propagate(start_state, bound_plane);
            if (!dest_state.isValid()) {
              // std::cout << "failed to propagate" << std::endl;
              continue;
            }
            const GlobalPoint&& dest_global_pos = dest_state.globalPosition();
            const LocalPoint&& local_point = eta_partition->toLocal(dest_global_pos);
            const LocalPoint local_point_2d{local_point.x(), local_point.y(), 0.0f};

            if (eta_partition->surface().bounds().inside(local_point_2d)) {
              //// PROPAGATED HIT ERROR EVALUATION
              // X,Y
              double xx = dest_state.curvilinearError().matrix()(3, 3);
              double yy = dest_state.curvilinearError().matrix()(4, 4);
              double xy = dest_state.curvilinearError().matrix()(4, 3);
              double dest_glob_error_x = sqrt(0.5 * (xx + yy - sqrt((xx - yy) * (xx - yy) + 4 * xy * xy)));
              double dest_glob_error_y = sqrt(0.5 * (xx + yy + sqrt((xx - yy) * (xx - yy) + 4 * xy * xy)));

              // R,Phi
              const LocalPoint&& dest_local_pos = eta_partition->toLocal(dest_global_pos);
              const LocalError&& dest_local_err = dest_state.localError().positionError();
              const GlobalError& dest_global_err =
                  ErrorFrameTransformer().transform(dest_local_err, eta_partition->surface());
              const double dest_global_r_err = std::sqrt(dest_global_err.rerr(dest_global_pos));
              const double dest_global_phi_err = std::sqrt(dest_global_err.phierr(dest_global_pos));

              ++nProp;

              propagated_muIdx.push_back(nMuons - 1);

              propagated_nME1hits.push_back(nME1_hits);
              propagated_nME2hits.push_back(nME2_hits);
              propagated_nME3hits.push_back(nME3_hits);
              propagated_nME4hits.push_back(nME4_hits);

              const auto& eta_partition_pos{eta_partition->position()};
              const auto& eta_partition_surf{eta_partition->surface()};
              propagated_EtaPartition_centerX.push_back(eta_partition_pos.x());
              propagated_EtaPartition_centerY.push_back(eta_partition_pos.y());
              propagated_EtaPartition_rMin.push_back(eta_partition_surf.rSpan().first);
              propagated_EtaPartition_rMax.push_back(eta_partition_surf.rSpan().second);
              propagated_EtaPartition_phiMin.push_back(eta_partition_surf.phiSpan().first);
              propagated_EtaPartition_phiMax.push_back(eta_partition_surf.phiSpan().second);

              propagatedGlb_x.push_back(dest_global_pos.x());
              propagatedGlb_y.push_back(dest_global_pos.y());
              propagatedGlb_z.push_back(dest_global_pos.z());
              propagatedGlb_r.push_back(dest_global_pos.perp());
              propagatedGlb_phi.push_back(dest_global_pos.phi());

              const auto dest_local_dir{dest_state.localDirection()};
              propagatedLoc_x.push_back(dest_local_pos.x());
              propagatedLoc_y.push_back(dest_local_pos.y());
              propagatedLoc_z.push_back(dest_local_pos.z());
              propagatedLoc_r.push_back(dest_local_pos.perp());
              propagatedLoc_phi.push_back(dest_local_pos.phi());
              propagatedLoc_dirX.push_back(dest_local_dir.x());
              propagatedLoc_dirY.push_back(dest_local_dir.y());
              propagatedLoc_dirZ.push_back(dest_local_dir.z());

              propagatedLoc_errX.push_back(dest_local_err.xx());
              propagatedLoc_errY.push_back(dest_local_err.yy());

              propagatedGlb_errX.push_back(dest_glob_error_x);
              propagatedGlb_errY.push_back(dest_glob_error_y);
              propagatedGlb_rerr.push_back(dest_global_r_err);
              propagatedGlb_phierr.push_back(dest_global_phi_err);

              propagated_region.push_back(gem_id.region());
              propagated_layer.push_back(gem_id.layer());
              propagated_chamber.push_back(gem_id.chamber());
              propagated_etaP.push_back(gem_id.roll());

              propagated_isinsideout.push_back(is_insideout);
              propagated_isincoming.push_back(is_incoming);

            }   //propagation is inside boundaries
          }     //loop on EtaPartitions
        }       //is_csc therefore perform propagation
      } else {  //!muon.outerTrack().isNull()
        innermost_x.push_back(DEFAULT_DOUBLE_VAL);
        innermost_y.push_back(DEFAULT_DOUBLE_VAL);
        innermost_z.push_back(DEFAULT_DOUBLE_VAL);
        outermost_x.push_back(DEFAULT_DOUBLE_VAL);
        outermost_y.push_back(DEFAULT_DOUBLE_VAL);
        outermost_z.push_back(DEFAULT_DOUBLE_VAL);
      }
      isCSC.push_back(is_csc);
      isME11.push_back(is_me11);

    }  //loop on reco muons
  }

  auto table = std::make_unique<nanoaod::FlatTable>(nMuons, m_name, false, true);

  //table->setDoc("RECO muon information");

  addColumn(table, "innermost_x", innermost_x, "");
  addColumn(table, "innermost_y", innermost_y, "");
  addColumn(table, "innermost_z", innermost_z, "");

  addColumn(table, "outermost_x", outermost_x, "");
  addColumn(table, "outermost_y", outermost_y, "");
  addColumn(table, "outermost_z", outermost_z, "");
  ev.put(std::move(table));

  if (m_fillPropagated) {
    auto tabProp = std::make_unique<nanoaod::FlatTable>(nProp, m_name + "_propagated", false, false);

    addColumn(tabProp, "propagated_muIdx", propagated_muIdx, "");

    addColumn(tabProp,
              "propagated_nME1hits",
              propagated_nME1hits,
              "number of hits in the CSC ME1 station"
              "in the STA muon track extrapolated to GE11");
    addColumn(tabProp,
              "propagated_nME2hits",
              propagated_nME2hits,
              "number of hits in the CSC ME2 station"
              "in the STA muon track extrapolated to GE11");
    addColumn(tabProp,
              "propagated_nME3hits",
              propagated_nME3hits,
              "number of hits in the CSC ME3 station"
              "in the STA muon track extrapolated to GE11");
    addColumn(tabProp,
              "propagated_nME4hits",
              propagated_nME4hits,
              "number of hits in the CSC ME4 station"
              "in the STA muon track extrapolated to GE11");

    addColumn(
        tabProp, "propagated_isincoming", propagated_isincoming, "bool, condition on the muon STA track direction");
    addColumn(
        tabProp, "propagated_isinsideout", propagated_isinsideout, "bool, condition on the muon STA track direction");
    addColumn(tabProp,
              "propagated_region",
              propagated_region,
              "GE11 region where the extrapolated muon track falls"
              "<br />(int, positive endcap: +1, negative endcap: -1");
    addColumn(tabProp,
              "propagated_layer",
              propagated_layer,
              "GE11 layer where the extrapolated muon track falls"
              "<br />(int, layer1: 1, layer2: 2");
    addColumn(tabProp,
              "propagated_chamber",
              propagated_chamber,
              "GE11 superchamber where the extrapolated muon track falls"
              "<br />(int, chambers numbered from 0 to 35");
    addColumn(tabProp,
              "propagated_etaP",
              propagated_etaP,
              "GE11 eta partition where the extrapolated muon track falls"
              "<br />(int, partitions numbered from 1 to 8");

    addColumn(tabProp,
              "propagatedLoc_x",
              propagatedLoc_x,
              "expected position of muon track extrapolated to GE11 surface"
              "<br />(float, local layer x coordinates, cm)");
    addColumn(tabProp,
              "propagatedLoc_y",
              propagatedLoc_y,
              "expected position of muon track extrapolated to GE11 surface"
              "<br />(float, local layer y coordinates, cm)");
    addColumn(tabProp,
              "propagatedLoc_z",
              propagatedLoc_z,
              "expected position of muon track extrapolated to GE11 surface"
              "<br />(float, local layer z coordinates, cm)");
    addColumn(tabProp,
              "propagatedLoc_r",
              propagatedLoc_r,
              "expected position of muon track extrapolated to GE11 surface"
              "<br />(float, local layer radial coordinate, cm)");
    addColumn(tabProp,
              "propagatedLoc_phi",
              propagatedLoc_phi,
              "expected position of muon track extrapolated to GE11 surface"
              "<br />(float, local layer phi coordinates, rad)");

    addColumn(tabProp,
              "propagatedLoc_dirX",
              propagatedLoc_dirX,
              "direction cosine of angle between local x axis and GE11 plane"
              "<br />(float, dir. cosine)");
    addColumn(tabProp,
              "propagatedLoc_dirY",
              propagatedLoc_dirY,
              "direction cosine of angle between local y axis and GE11 plane"
              "<br />(float, dir. cosine)");
    addColumn(tabProp,
              "propagatedLoc_dirZ",
              propagatedLoc_dirZ,
              "direction cosine of angle between local z axis and GE11 plane"
              "<br />(float, dir. cosine)");

    addColumn(tabProp,
              "propagatedLoc_errX",
              propagatedLoc_errX,
              "uncertainty on expected position of muon track extrapolated to GE11 surface"
              "<br />(float, local layer x coordinates, cm)");
    addColumn(tabProp,
              "propagatedLoc_errY",
              propagatedLoc_errY,
              "uncertainty on expected position of muon track extrapolated to GE11 surface"
              "<br />(float, local layer y coordinates, cm)");

    addColumn(tabProp,
              "propagatedGlb_x",
              propagatedGlb_x,
              "expected position of muon track extrapolated to GE11 surface"
              "<br />(float, global x coordinates, cm)");
    addColumn(tabProp,
              "propagatedGlb_y",
              propagatedGlb_y,
              "expected position of muon track extrapolated to GE11 surface"
              "<br />(float, global y coordinates, cm)");
    addColumn(tabProp,
              "propagatedGlb_z",
              propagatedGlb_z,
              "expected position of muon track extrapolated to GE11 surface"
              "<br />(float, global z coordinates, cm)");
    addColumn(tabProp,
              "propagatedGlb_r",
              propagatedGlb_r,
              "expected position of muon track extrapolated to GE11 surface"
              "<br />(float, global radial (r) coordinates, cm)");
    addColumn(tabProp,
              "propagatedGlb_phi",
              propagatedGlb_phi,
              "expected position of muon track extrapolated to GE11 surface"
              "<br />(float, global phi coordinates, rad)");
    addColumn(tabProp,
              "propagatedGlb_errX",
              propagatedGlb_errX,
              "uncertainty on position of muon track extrapolated to GE11 surface"
              "<br />(float, global x coordinates, cm)");
    addColumn(tabProp,
              "propagatedGlb_errY",
              propagatedGlb_errY,
              "uncertainty on position of muon track extrapolated to GE11 surface"
              "<br />(float, global y coordinates, cm)");
    addColumn(tabProp,
              "propagatedGlb_rerr",
              propagatedGlb_rerr,
              "uncertainty on position of muon track extrapolated to GE11 surface"
              "<br />(float, global radial (r) coordinates, cm)");
    addColumn(tabProp,
              "propagatedGlb_phierr",
              propagatedGlb_phierr,
              "uncertainty on position of muon track extrapolated to GE11 surface"
              "<br />(float, global phi coordinates, rad)");

    addColumn(tabProp,
              "propagated_EtaPartition_centerX",
              propagated_EtaPartition_centerX,
              "global X coordinate of the center of the etaPartition"
              "<br />where the extrapolated muon track position falls"
              "<br />(float, global x coordinates, cm)");
    addColumn(tabProp,
              "propagated_EtaPartition_centerY",
              propagated_EtaPartition_centerY,
              "global Y coordinate of the center of the etaPartition"
              "<br />where the extrapolated muon track position falls"
              "<br />(float, global x coordinates, cm)");
    addColumn(tabProp,
              "propagated_EtaPartition_phiMax",
              propagated_EtaPartition_phiMax,
              "upper edge in phi global coordinates of the etaPartition"
              "<br />where the extrapolated muon track position falls"
              "<br />(float, global phi coordinates, rad)");
    addColumn(tabProp,
              "propagated_EtaPartition_phiMin",
              propagated_EtaPartition_phiMin,
              "lower edge in phi global coordinates of the etaPartition"
              "<br />where the extrapolated muon track position falls"
              "<br />(float, global phi coordinates, rad)");
    addColumn(tabProp,
              "propagated_EtaPartition_rMax",
              propagated_EtaPartition_rMax,
              "upper edge in r global coordinates of the etaPartition"
              "<br />where the extrapolated muon track position falls"
              "<br />(float, global radial (r) coordinates, cm)");
    addColumn(tabProp,
              "propagated_EtaPartition_rMin",
              propagated_EtaPartition_rMin,
              "lower edge in r global coordinates of the etaPartition"
              "<br />where the extrapolated muon track position falls"
              "<br />(float, global radial (r) coordinates, cm)");

    ev.put(std::move(tabProp), "propagated");
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuGEMMuonExtTableProducer);
