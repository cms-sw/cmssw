#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFit/interface/MkFitGeometry.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

namespace {
  using namespace mkfit;

  [[maybe_unused]] void partitionSeeds0(const TrackerInfo &trk_info,
                                        const TrackVec &in_seeds,
                                        const EventOfHits &eoh,
                                        IterationSeedPartition &part) {
    const size_t size = in_seeds.size();

    for (size_t i = 0; i < size; ++i) {
      const Track &S = in_seeds[i];

      const bool z_dir_pos = S.pz() > 0;

      const auto &hot = S.getLastHitOnTrack();
      const float eta = eoh[hot.layer].refHit(hot.index).eta();

      // Region to be defined by propagation / intersection tests
      TrackerInfo::EtaRegion reg;

      // Max eta used for region sorting
      constexpr float maxEta_regSort = 7.0;

      const LayerInfo &outer_brl = trk_info.outer_barrel_layer();

      // Define first (mkFit) layer IDs for each strip subdetector.
      constexpr int tib1_id = 4;
      constexpr int tob1_id = 10;
      constexpr int tecp1_id = 27;
      constexpr int tecn1_id = 54;

      const LayerInfo &tib1 = trk_info.layer(tib1_id);
      const LayerInfo &tob1 = trk_info.layer(tob1_id);

      const LayerInfo &tecp1 = trk_info.layer(tecp1_id);
      const LayerInfo &tecn1 = trk_info.layer(tecn1_id);

      const LayerInfo &tec_first = z_dir_pos ? tecp1 : tecn1;

      const float maxR = S.maxReachRadius();
      float z_at_maxr;

      bool can_reach_outer_brl = S.canReachRadius(outer_brl.rout());
      float z_at_outer_brl;
      bool misses_first_tec;
      if (can_reach_outer_brl) {
        z_at_outer_brl = S.zAtR(outer_brl.rout());
        if (z_dir_pos)
          misses_first_tec = z_at_outer_brl < tec_first.zmin();
        else
          misses_first_tec = z_at_outer_brl > tec_first.zmax();
      } else {
        z_at_maxr = S.zAtR(maxR);
        if (z_dir_pos)
          misses_first_tec = z_at_maxr < tec_first.zmin();
        else
          misses_first_tec = z_at_maxr > tec_first.zmax();
      }

      if (misses_first_tec) {
        reg = TrackerInfo::Reg_Barrel;
      } else {
        if ((S.canReachRadius(tib1.rin()) && tib1.is_within_z_limits(S.zAtR(tib1.rin()))) ||
            (S.canReachRadius(tob1.rin()) && tob1.is_within_z_limits(S.zAtR(tob1.rin())))) {
          reg = z_dir_pos ? TrackerInfo::Reg_Transition_Pos : TrackerInfo::Reg_Transition_Neg;
        } else {
          reg = z_dir_pos ? TrackerInfo::Reg_Endcap_Pos : TrackerInfo::Reg_Endcap_Neg;
        }
      }

      part.m_region[i] = reg;

      // TrackerInfo::EtaRegion is enum from 0 to 5 (Reg_Endcap_Neg,Reg_Transition_Neg,Reg_Barrel,Reg_Transition_Pos,Reg_Endcap_Pos)
      // Symmetrization around TrackerInfo::Reg_Barrel for sorting is required
      part.m_sort_score[i] = maxEta_regSort * (reg - TrackerInfo::Reg_Barrel) + eta;
    }
  }

  [[maybe_unused]] void partitionSeeds1(const TrackerInfo &trk_info,
                                        const TrackVec &in_seeds,
                                        const EventOfHits &eoh,
                                        IterationSeedPartition &part) {
    // Define first (mkFit) layer IDs for each strip subdetector.
    constexpr int tib1_id = 4;
    constexpr int tob1_id = 10;
    constexpr int tidp1_id = 21;
    constexpr int tidn1_id = 48;
    constexpr int tecp1_id = 27;
    constexpr int tecn1_id = 54;

    const LayerInfo &tib1 = trk_info.layer(tib1_id);
    const LayerInfo &tob1 = trk_info.layer(tob1_id);

    const LayerInfo &tidp1 = trk_info.layer(tidp1_id);
    const LayerInfo &tidn1 = trk_info.layer(tidn1_id);

    const LayerInfo &tecp1 = trk_info.layer(tecp1_id);
    const LayerInfo &tecn1 = trk_info.layer(tecn1_id);

    // Merge first two layers to account for mono/stereo coverage.
    // TrackerInfo could hold joint limits for sub-detectors.
    const auto &L = trk_info;
    const float tidp_rin = std::min(L[tidp1_id].rin(), L[tidp1_id + 1].rin());
    const float tidp_rout = std::max(L[tidp1_id].rout(), L[tidp1_id + 1].rout());
    const float tecp_rin = std::min(L[tecp1_id].rin(), L[tecp1_id + 1].rin());
    const float tecp_rout = std::max(L[tecp1_id].rout(), L[tecp1_id + 1].rout());
    const float tidn_rin = std::min(L[tidn1_id].rin(), L[tidn1_id + 1].rin());
    const float tidn_rout = std::max(L[tidn1_id].rout(), L[tidn1_id + 1].rout());
    const float tecn_rin = std::min(L[tecn1_id].rin(), L[tecn1_id + 1].rin());
    const float tecn_rout = std::max(L[tecn1_id].rout(), L[tecn1_id + 1].rout());

    // Bias towards more aggressive transition-region assignemnts.
    // With current tunning it seems to make things a bit worse.
    const float tid_z_extra = 0.0f;  // 5.0f;
    const float tec_z_extra = 0.0f;  // 10.0f;

    const size_t size = in_seeds.size();

    auto barrel_pos_check = [](const Track &S, float maxR, float rin, float zmax) -> bool {
      bool inside = maxR > rin && S.zAtR(rin) < zmax;
      return inside;
    };

    auto barrel_neg_check = [](const Track &S, float maxR, float rin, float zmin) -> bool {
      bool inside = maxR > rin && S.zAtR(rin) > zmin;
      return inside;
    };

    auto endcap_pos_check = [](const Track &S, float maxR, float rout, float rin, float zmin) -> bool {
      bool inside = maxR > rout ? S.zAtR(rout) > zmin : (maxR > rin && S.zAtR(maxR) > zmin);
      return inside;
    };

    auto endcap_neg_check = [](const Track &S, float maxR, float rout, float rin, float zmax) -> bool {
      bool inside = maxR > rout ? S.zAtR(rout) < zmax : (maxR > rin && S.zAtR(maxR) < zmax);
      return inside;
    };

    for (size_t i = 0; i < size; ++i) {
      const Track &S = in_seeds[i];

      const auto &hot = S.getLastHitOnTrack();
      const float eta = eoh[hot.layer].refHit(hot.index).eta();

      // Region to be defined by propagation / intersection tests
      TrackerInfo::EtaRegion reg;

      // Max eta used for region sorting
      constexpr float maxEta_regSort = 7.0;

      const bool z_dir_pos = S.pz() > 0;
      const float maxR = S.maxReachRadius();

      if (z_dir_pos) {
        const bool in_tib = barrel_pos_check(S, maxR, tib1.rin(), tib1.zmax());
        const bool in_tob = barrel_pos_check(S, maxR, tob1.rin(), tob1.zmax());

        if (!in_tib && !in_tob) {
          reg = TrackerInfo::Reg_Endcap_Pos;
        } else {
          const bool in_tid = endcap_pos_check(S, maxR, tidp_rout, tidp_rin, tidp1.zmin() - tid_z_extra);
          const bool in_tec = endcap_pos_check(S, maxR, tecp_rout, tecp_rin, tecp1.zmin() - tec_z_extra);

          if (!in_tid && !in_tec) {
            reg = TrackerInfo::Reg_Barrel;
          } else {
            reg = TrackerInfo::Reg_Transition_Pos;
          }
        }
      } else {
        const bool in_tib = barrel_neg_check(S, maxR, tib1.rin(), tib1.zmin());
        const bool in_tob = barrel_neg_check(S, maxR, tob1.rin(), tob1.zmin());

        if (!in_tib && !in_tob) {
          reg = TrackerInfo::Reg_Endcap_Neg;
        } else {
          const bool in_tid = endcap_neg_check(S, maxR, tidn_rout, tidn_rin, tidn1.zmax() + tid_z_extra);
          const bool in_tec = endcap_neg_check(S, maxR, tecn_rout, tecn_rin, tecn1.zmax() + tec_z_extra);

          if (!in_tid && !in_tec) {
            reg = TrackerInfo::Reg_Barrel;
          } else {
            reg = TrackerInfo::Reg_Transition_Neg;
          }
        }
      }

      part.m_region[i] = reg;

      // TrackerInfo::EtaRegion is enum from 0 to 5 (Reg_Endcap_Neg,Reg_Transition_Neg,Reg_Barrel,Reg_Transition_Pos,Reg_Endcap_Pos)
      // Symmetrization around TrackerInfo::Reg_Barrel for sorting is required
      part.m_sort_score[i] = maxEta_regSort * (reg - TrackerInfo::Reg_Barrel) + eta;
    }
  }
}  // namespace

class MkFitIterationConfigESProducer : public edm::ESProducer {
public:
  MkFitIterationConfigESProducer(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::unique_ptr<mkfit::IterationConfig> produce(const TrackerRecoGeometryRecord &iRecord);

private:
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> geomToken_;
  const std::string configFile_;
  const float minPtCut_;
};

MkFitIterationConfigESProducer::MkFitIterationConfigESProducer(const edm::ParameterSet &iConfig)
    : geomToken_{setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName")).consumes()},
      configFile_{iConfig.getParameter<edm::FileInPath>("config").fullPath()},
      minPtCut_{(float)iConfig.getParameter<double>("minPt")} {}

void MkFitIterationConfigESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName")->setComment("Product label");
  desc.add<edm::FileInPath>("config")->setComment("Path to the JSON file for the mkFit configuration parameters");
  desc.add<double>("minPt", 0.0)->setComment("min pT cut applied during track building");
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<mkfit::IterationConfig> MkFitIterationConfigESProducer::produce(
    const TrackerRecoGeometryRecord &iRecord) {
  mkfit::ConfigJson cj;
  auto it_conf = cj.load_File(configFile_);
  it_conf->m_params.minPtCut = minPtCut_;
  it_conf->m_backward_params.minPtCut = minPtCut_;
  it_conf->m_partition_seeds = partitionSeeds1;
  return it_conf;
}

DEFINE_FWK_EVENTSETUP_MODULE(MkFitIterationConfigESProducer);
