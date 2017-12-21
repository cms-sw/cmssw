#ifndef RecoBTag_DeepFlavour_deep_helpers_h
#define RecoBTag_DeepFlavour_deep_helpers_h

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

#include "TrackingTools/IPTools/interface/IPTools.h"

#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
//#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"

namespace btagbtvdeep {

  // remove infs and NaNs with value  (adapted from DeepNTuples)
  inline const float catch_infs(const float in,
				const float replace_value);

  // remove infs/NaN and bound (adapted from DeepNTuples) 
  inline const float catch_infs_and_bound(const float in,
                                          const float replace_value,
                                          const float lowerbound,
                                          const float upperbound,
                                          const float offset=0.,
                                          const bool use_offsets = true);

  // 2D distance between SV and PV (adapted from DeepNTuples)
  Measurement1D vertexDxy(const reco::VertexCompositePtrCandidate &svcand, const reco::Vertex &pv);

  //3D distance between SV and PV (adapted from DeepNTuples)
  Measurement1D vertexD3d(const reco::VertexCompositePtrCandidate &svcand, const reco::Vertex &pv);
 
  // dot product between SV and PV (adapted from DeepNTuples)
  float vertexDdotP(const reco::VertexCompositePtrCandidate &sv, const reco::Vertex &pv);  

  // helper to order vertices by significance (adapted from DeepNTuples)
  template < typename SVType, typename PVType>
  bool sv_vertex_comparator(const SVType & sva, const SVType & svb, const PVType & pv);    

  // write tagging variables to vector (adapted from DeepNTuples)
  template <typename T>
  int dump_vector(reco::TaggingVariableList& from, T* to,
                  reco::btau::TaggingVariableName name, const size_t max);

  // compute minimum dr between SVs and a candidate (from DeepNTuples, now polymorphic)
  float mindrsvpfcand(const std::vector<reco::VertexCompositePtrCandidate> & svs, 
                      const reco::Candidate* cand, float mindr=0.4);
}
#endif //RecoBTag_DeepFlavour_deep_helpers_h
