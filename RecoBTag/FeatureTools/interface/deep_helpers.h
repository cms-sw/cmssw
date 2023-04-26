#ifndef RecoBTag_FeatureTools_deep_helpers_h
#define RecoBTag_FeatureTools_deep_helpers_h

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
//#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <nlohmann/json.hpp>

namespace btagbtvdeep {

  // remove infs and NaNs with value  (adapted from DeepNTuples)
  const float catch_infs(const float in, const float replace_value = 0.);

  // remove infs/NaN and bound (adapted from DeepNTuples)
  const float catch_infs_and_bound(const float in,
                                   const float replace_value,
                                   const float lowerbound,
                                   const float upperbound,
                                   const float offset = 0.,
                                   const bool use_offsets = true);

  // 2D distance between SV and PV (adapted from DeepNTuples)
  Measurement1D vertexDxy(const reco::VertexCompositePtrCandidate &svcand, const reco::Vertex &pv);

  //3D distance between SV and PV (adapted from DeepNTuples)
  Measurement1D vertexD3d(const reco::VertexCompositePtrCandidate &svcand, const reco::Vertex &pv);

  // dot product between SV and PV (adapted from DeepNTuples)
  float vertexDdotP(const reco::VertexCompositePtrCandidate &sv, const reco::Vertex &pv);

  // helper to order vertices by significance (adapted from DeepNTuples)
  template <typename SVType, typename PVType>
  bool sv_vertex_comparator(const SVType &sva, const SVType &svb, const PVType &pv) {
    auto adxy = vertexDxy(sva, pv);
    auto bdxy = vertexDxy(svb, pv);
    float aval = adxy.value();
    float bval = bdxy.value();
    float aerr = adxy.error();
    float berr = bdxy.error();

    float asig = catch_infs(aval / aerr, 0.);
    float bsig = catch_infs(bval / berr, 0.);
    return bsig < asig;
  }

  // write tagging variables to vector (adapted from DeepNTuples)
  template <typename T>
  int dump_vector(reco::TaggingVariableList &from, T *to, reco::btau::TaggingVariableName name, const size_t max) {
    std::vector<T> vals = from.getList(name, false);
    size_t size = std::min(vals.size(), max);
    if (size > 0) {
      for (size_t i = 0; i < vals.size(); i++) {
        to[i] = catch_infs(vals[i], -0.1);
      }
    }
    return size;
  }

  // compute minimum dr between SVs and a candidate (from DeepNTuples, now polymorphic)
  float mindrsvpfcand(const std::vector<reco::VertexCompositePtrCandidate> &svs,
                      const reco::Candidate *cand,
                      float mindr = 0.4);

  // compute minimum distance between SVs and a candidate (from DeepNTuples, now polymorphic)
  float mindistsvpfcand(const std::vector<reco::VertexCompositePtrCandidate> &svs, const reco::TransientTrack track);
    
  // mimic the calculation in PackedCandidate
  float vtx_ass_from_pfcand(const reco::PFCandidate &pfcand, int pv_ass_quality, const reco::VertexRef &pv);
  float quality_from_pfcand(const reco::PFCandidate &pfcand);
  float lost_inner_hits_from_pfcand(const reco::PFCandidate &pfcand);

  std::pair<float, float> getDRSubjetFeatures(const reco::Jet &jet, const reco::Candidate *cand);

  // struct to hold preprocessing parameters
  struct PreprocessParams {
    struct VarInfo {
      VarInfo() {}
      VarInfo(float median, float norm_factor, float replace_inf_value, float lower_bound, float upper_bound, float pad)
          : center(median),
            norm_factor(norm_factor),
            replace_inf_value(replace_inf_value),
            lower_bound(lower_bound),
            upper_bound(upper_bound),
            pad(pad) {}
      float center = 0;
      float norm_factor = 1;
      float replace_inf_value = 0;
      float lower_bound = -5;
      float upper_bound = 5;
      float pad = 0;
    };

    unsigned min_length = 0;
    unsigned max_length = 0;
    std::vector<std::string> var_names;
    std::unordered_map<std::string, VarInfo> var_info_map;

    VarInfo info(const std::string &name) const { return var_info_map.at(name); }
  };

  int center_norm_pad(const std::vector<float> &input,
                      float center,
                      float scale,
                      unsigned min_length,
                      unsigned max_length,
                      std::vector<float> &datavec,
                      int startval,
                      float pad_value = 0,
                      float replace_inf_value = 0,
                      float min = 0,
                      float max = -1);

  int center_norm_pad_halfRagged(const std::vector<float> &input,
                                 float center,
                                 float scale,
                                 unsigned target_length,
                                 std::vector<float> &datavec,
                                 int startval,
                                 float pad_value = 0,
                                 float replace_inf_value = 0,
                                 float min = 0,
                                 float max = -1);

  void ParticleNetConstructor(const edm::ParameterSet &Config_,
                              bool doExtra,
                              std::vector<std::string> &input_names_,
                              std::unordered_map<std::string, PreprocessParams> &prep_info_map_,
                              std::vector<std::vector<int64_t>> &input_shapes_,
                              std::vector<unsigned> &input_sizes_,
                              cms::Ort::FloatArrays *data_);

}  // namespace btagbtvdeep
#endif  //RecoBTag_FeatureTools_deep_helpers_h
