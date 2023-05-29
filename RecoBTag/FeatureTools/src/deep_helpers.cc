#include "RecoBTag/FeatureTools/interface/deep_helpers.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalTrajectoryExtrapolatorToLine.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TLorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
namespace btagbtvdeep {

  constexpr static int qualityMap[8] = {1, 0, 1, 1, 4, 4, 5, 6};

  enum qualityFlagsShiftsAndMasks {
    assignmentQualityMask = 0x7,
    assignmentQualityShift = 0,
    trackHighPurityMask = 0x8,
    trackHighPurityShift = 3,
    lostInnerHitsMask = 0x30,
    lostInnerHitsShift = 4,
    muonFlagsMask = 0x0600,
    muonFlagsShift = 9
  };

  // remove infs and NaNs with value  (adapted from DeepNTuples)
  const float catch_infs(const float in, const float replace_value) {
    if (edm::isNotFinite(in))
      return replace_value;
    if (in < -1e32 || in > 1e32)
      return replace_value;
    return in;
  }

  // remove infs/NaN and bound (adapted from DeepNTuples)
  const float catch_infs_and_bound(const float in,
                                   const float replace_value,
                                   const float lowerbound,
                                   const float upperbound,
                                   const float offset,
                                   const bool use_offsets) {
    float withoutinfs = catch_infs(in, replace_value);
    if (withoutinfs + offset < lowerbound)
      return lowerbound;
    if (withoutinfs + offset > upperbound)
      return upperbound;
    if (use_offsets)
      withoutinfs += offset;
    return withoutinfs;
  }

  // 2D distance between SV and PV (adapted from DeepNTuples)
  Measurement1D vertexDxy(const reco::VertexCompositePtrCandidate &svcand, const reco::Vertex &pv) {
    VertexDistanceXY dist;
    reco::Vertex::CovarianceMatrix csv;
    svcand.fillVertexCovariance(csv);
    reco::Vertex svtx(svcand.vertex(), csv);
    return dist.distance(svtx, pv);
  }

  //3D distance between SV and PV (adapted from DeepNTuples)
  Measurement1D vertexD3d(const reco::VertexCompositePtrCandidate &svcand, const reco::Vertex &pv) {
    VertexDistance3D dist;
    reco::Vertex::CovarianceMatrix csv;
    svcand.fillVertexCovariance(csv);
    reco::Vertex svtx(svcand.vertex(), csv);
    return dist.distance(svtx, pv);
  }

  // dot product between SV and PV (adapted from DeepNTuples)
  float vertexDdotP(const reco::VertexCompositePtrCandidate &sv, const reco::Vertex &pv) {
    reco::Candidate::Vector p = sv.momentum();
    reco::Candidate::Vector d(sv.vx() - pv.x(), sv.vy() - pv.y(), sv.vz() - pv.z());
    return p.Unit().Dot(d.Unit());
  }

  // compute minimum dr between SVs and a candidate (from DeepNTuples, now polymorphic)
  float mindrsvpfcand(const std::vector<reco::VertexCompositePtrCandidate> &svs,
                      const reco::Candidate *cand,
                      float mindr) {
    for (unsigned int i0 = 0; i0 < svs.size(); ++i0) {
      float tempdr = reco::deltaR(svs[i0], *cand);
      if (tempdr < mindr) {
        mindr = tempdr;
      }
    }
    return mindr;
  }

  // compute minimum distance between SVs and a candidate (from DeepNTuples, now polymorphic)
  float mindistsvpfcand(const std::vector<reco::VertexCompositePtrCandidate> &svs, const reco::TransientTrack track) {
    float mindist_ = 999.999;
    float out_dist = 0.0;
    for (unsigned int i = 0; i < svs.size(); ++i) {
      if (!track.isValid()) {
        continue;
      }
      reco::Vertex::CovarianceMatrix csv;
      svs[i].fillVertexCovariance(csv);
      reco::Vertex vertex(svs[i].vertex(), csv);
      if (!vertex.isValid()) {
        continue;
      }

      GlobalVector direction(svs[i].px(), svs[i].py(), svs[i].pz());

      AnalyticalImpactPointExtrapolator extrapolator(track.field());
      TrajectoryStateOnSurface tsos =
          extrapolator.extrapolate(track.impactPointState(), RecoVertex::convertPos(vertex.position()));

      VertexDistance3D dist;

      if (!tsos.isValid()) {
        continue;
      }
      GlobalPoint refPoint = tsos.globalPosition();
      GlobalError refPointErr = tsos.cartesianError().position();
      GlobalPoint vertexPosition = RecoVertex::convertPos(vertex.position());
      GlobalError vertexPositionErr = RecoVertex::convertError(vertex.error());

      std::pair<bool, Measurement1D> result(
          true, dist.distance(VertexState(vertexPosition, vertexPositionErr), VertexState(refPoint, refPointErr)));
      if (!result.first) {
        continue;
      }

      GlobalPoint impactPoint = tsos.globalPosition();
      GlobalVector IPVec(impactPoint.x() - vertex.x(), impactPoint.y() - vertex.y(), impactPoint.z() - vertex.z());
      double prod = IPVec.dot(direction);
      double sign = (prod >= 0) ? 1. : -1.;

      if (result.second.value() < mindist_) {
        out_dist = sign * result.second.value();
        mindist_ = result.second.value();
      }
    }
    return out_dist;
  }

  // instantiate template
  template bool sv_vertex_comparator<reco::VertexCompositePtrCandidate, reco::Vertex>(
      const reco::VertexCompositePtrCandidate &, const reco::VertexCompositePtrCandidate &, const reco::Vertex &);

  float vtx_ass_from_pfcand(const reco::PFCandidate &pfcand, int pv_ass_quality, const reco::VertexRef &pv) {
    float vtx_ass = pat::PackedCandidate::PVAssociationQuality(qualityMap[pv_ass_quality]);
    if (pfcand.trackRef().isNonnull() && pv->trackWeight(pfcand.trackRef()) > 0.5 && pv_ass_quality == 7) {
      vtx_ass = pat::PackedCandidate::UsedInFitTight;
    }
    return vtx_ass;
  }

  float quality_from_pfcand(const reco::PFCandidate &pfcand) {
    const auto &pseudo_track = (pfcand.bestTrack()) ? *pfcand.bestTrack() : reco::Track();
    // conditions from PackedCandidate producer
    bool highPurity = pfcand.trackRef().isNonnull() && pseudo_track.quality(reco::Track::highPurity);
    // do same bit operations than in PackedCandidate
    uint16_t qualityFlags = 0;
    qualityFlags = (qualityFlags & ~trackHighPurityMask) | ((highPurity << trackHighPurityShift) & trackHighPurityMask);
    bool isHighPurity = (qualityFlags & trackHighPurityMask) >> trackHighPurityShift;
    // to do as in TrackBase
    uint8_t quality = (1 << reco::TrackBase::loose);
    if (isHighPurity) {
      quality |= (1 << reco::TrackBase::highPurity);
    }
    return quality;
  }

  float lost_inner_hits_from_pfcand(const reco::PFCandidate &pfcand) {
    const auto &pseudo_track = (pfcand.bestTrack()) ? *pfcand.bestTrack() : reco::Track();
    // conditions from PackedCandidate producer
    bool highPurity = pfcand.trackRef().isNonnull() && pseudo_track.quality(reco::Track::highPurity);
    // do same bit operations than in PackedCandidate
    uint16_t qualityFlags = 0;
    qualityFlags = (qualityFlags & ~trackHighPurityMask) | ((highPurity << trackHighPurityShift) & trackHighPurityMask);
    return int16_t((qualityFlags & lostInnerHitsMask) >> lostInnerHitsShift) - 1;
  }

  std::pair<float, float> getDRSubjetFeatures(const reco::Jet &jet, const reco::Candidate *cand) {
    const auto *patJet = dynamic_cast<const pat::Jet *>(&jet);
    std::pair<float, float> features;
    // Do Subjets
    if (patJet) {
      if (patJet->nSubjetCollections() > 0) {
        auto subjets = patJet->subjets();
        std::nth_element(
            subjets.begin(),
            subjets.begin() + 1,
            subjets.end(),
            [](const edm::Ptr<pat::Jet> &p1, const edm::Ptr<pat::Jet> &p2) { return p1->pt() > p2->pt(); });
        features.first = !subjets.empty() ? reco::deltaR(*cand, *subjets[0]) : -1;
        features.second = subjets.size() > 1 ? reco::deltaR(*cand, *subjets[1]) : -1;
      } else {
        features.first = -1;
        features.second = -1;
      }
    } else {
      features.first = -1;
      features.second = -1;
    }
    return features;
  }

  int center_norm_pad(const std::vector<float> &input,
                      float center,
                      float norm_factor,
                      unsigned min_length,
                      unsigned max_length,
                      std::vector<float> &datavec,
                      int startval,
                      float pad_value,
                      float replace_inf_value,
                      float min,
                      float max) {
    // do variable shifting/scaling/padding/clipping in one go

    assert(min <= pad_value && pad_value <= max);
    assert(min_length <= max_length);

    unsigned target_length = std::clamp((unsigned)input.size(), min_length, max_length);
    for (unsigned i = 0; i < target_length; ++i) {
      if (i < input.size()) {
        datavec[i + startval] = std::clamp((catch_infs(input[i], replace_inf_value) - center) * norm_factor, min, max);
      } else {
        datavec[i + startval] = pad_value;
      }
    }
    return target_length;
  }

  int center_norm_pad_halfRagged(const std::vector<float> &input,
                                 float center,
                                 float norm_factor,
                                 unsigned target_length,
                                 std::vector<float> &datavec,
                                 int startval,
                                 float pad_value,
                                 float replace_inf_value,
                                 float min,
                                 float max) {
    // do variable shifting/scaling/padding/clipping in one go

    assert(min <= pad_value && pad_value <= max);

    for (unsigned i = 0; i < std::min(static_cast<unsigned int>(input.size()), target_length); ++i) {
      datavec.push_back(std::clamp((catch_infs(input[i], replace_inf_value) - center) * norm_factor, min, max));
    }
    if (input.size() < target_length)
      datavec.insert(datavec.end(), target_length - input.size(), pad_value);

    return target_length;
  }

  void ParticleNetConstructor(const edm::ParameterSet &Config_,
                              bool doExtra,
                              std::vector<std::string> &input_names_,
                              std::unordered_map<std::string, PreprocessParams> &prep_info_map_,
                              std::vector<std::vector<int64_t>> &input_shapes_,
                              std::vector<unsigned> &input_sizes_,
                              cms::Ort::FloatArrays *data_) {
    // load preprocessing info
    auto json_path = Config_.getParameter<std::string>("preprocess_json");
    if (!json_path.empty()) {
      // use preprocessing json file if available
      std::ifstream ifs(edm::FileInPath(json_path).fullPath());
      nlohmann::json js = nlohmann::json::parse(ifs);
      js.at("input_names").get_to(input_names_);
      for (const auto &group_name : input_names_) {
        const auto &group_pset = js.at(group_name);
        auto &prep_params = prep_info_map_[group_name];
        group_pset.at("var_names").get_to(prep_params.var_names);
        if (group_pset.contains("var_length")) {
          prep_params.min_length = group_pset.at("var_length");
          prep_params.max_length = prep_params.min_length;
        } else {
          prep_params.min_length = group_pset.at("min_length");
          prep_params.max_length = group_pset.at("max_length");
          input_shapes_.push_back({1, (int64_t)prep_params.var_names.size(), -1});
        }
        const auto &var_info_pset = group_pset.at("var_infos");
        for (const auto &var_name : prep_params.var_names) {
          const auto &var_pset = var_info_pset.at(var_name);
          double median = var_pset.at("median");
          double norm_factor = var_pset.at("norm_factor");
          double replace_inf_value = var_pset.at("replace_inf_value");
          double lower_bound = var_pset.at("lower_bound");
          double upper_bound = var_pset.at("upper_bound");
          double pad = var_pset.contains("pad") ? double(var_pset.at("pad")) : 0;
          prep_params.var_info_map[var_name] =
              PreprocessParams::VarInfo(median, norm_factor, replace_inf_value, lower_bound, upper_bound, pad);
        }

        if (doExtra && data_ != nullptr) {
          // create data storage with a fixed size vector initialized w/ 0
          const auto &len = input_sizes_.emplace_back(prep_params.max_length * prep_params.var_names.size());
          data_->emplace_back(len, 0);
        }
      }
    } else {
      // otherwise use the PSet in the python config file
      const auto &prep_pset = Config_.getParameterSet("preprocessParams");
      input_names_ = prep_pset.getParameter<std::vector<std::string>>("input_names");
      for (const auto &group_name : input_names_) {
        const edm::ParameterSet &group_pset = prep_pset.getParameterSet(group_name);
        auto &prep_params = prep_info_map_[group_name];
        prep_params.var_names = group_pset.getParameter<std::vector<std::string>>("var_names");
        prep_params.min_length = group_pset.getParameter<unsigned>("var_length");
        prep_params.max_length = prep_params.min_length;
        const auto &var_info_pset = group_pset.getParameterSet("var_infos");
        for (const auto &var_name : prep_params.var_names) {
          const edm::ParameterSet &var_pset = var_info_pset.getParameterSet(var_name);
          double median = var_pset.getParameter<double>("median");
          double norm_factor = var_pset.getParameter<double>("norm_factor");
          double replace_inf_value = var_pset.getParameter<double>("replace_inf_value");
          double lower_bound = var_pset.getParameter<double>("lower_bound");
          double upper_bound = var_pset.getParameter<double>("upper_bound");
          prep_params.var_info_map[var_name] =
              PreprocessParams::VarInfo(median, norm_factor, replace_inf_value, lower_bound, upper_bound, 0);
        }

        if (doExtra && data_ != nullptr) {
          // create data storage with a fixed size vector initialized w/ 0
          const auto &len = input_sizes_.emplace_back(prep_params.max_length * prep_params.var_names.size());
          data_->emplace_back(len, 0);
        }
      }
    }
  }

}  // namespace btagbtvdeep
