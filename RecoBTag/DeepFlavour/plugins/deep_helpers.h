#ifndef RecoBTag_DeepFlavour_deep_helpers_h
#define RecoBTag_DeepFlavour_deep_helpers_h

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/IPTools/interface/IPTools.h"



namespace btagbtvdeep {

  // remove infs and NaNs with value  (adapted from DeepNTuples)
  inline const float catch_infs(const float in,
                                 const float replace_value) {
    if(in==in){ // check if NaN 
      if(std::isinf(in))
        return replace_value;
      else if(in < -1e32 || in > 1e32)
        return replace_value;
      return in;
    }
    return replace_value;
  } 

  // remove infs/NaN and bound (adapted from DeepNTuples)
  inline const float catch_infs_and_bound(const float in,
                                          const float replace_value,
                                          const float lowerbound,
                                          const float upperbound,
                                          const float offset=0.,
                                          const bool use_offsets = true){
        float withoutinfs=catch_infs(in,replace_value);
        if(withoutinfs+offset<lowerbound) return lowerbound;
        if(withoutinfs+offset>upperbound) return upperbound;
        if(use_offsets)
            withoutinfs+=offset;
        return withoutinfs;
  }


  // 2D distance between SV and PV (adapted from DeepNTuples)
  Measurement1D vertexDxy(const reco::VertexCompositePtrCandidate &svcand, const reco::Vertex &pv)  {
    VertexDistanceXY dist;
    reco::Vertex::CovarianceMatrix csv; svcand.fillVertexCovariance(csv);
    reco::Vertex svtx(svcand.vertex(), csv);
    return dist.distance(svtx, pv);
  }

  //3D distance between SV and PV (adapted from DeepNTuples)
  Measurement1D vertexD3d(const reco::VertexCompositePtrCandidate &svcand, const reco::Vertex &pv)  {
    VertexDistance3D dist;
    reco::Vertex::CovarianceMatrix csv; svcand.fillVertexCovariance(csv);
    reco::Vertex svtx(svcand.vertex(), csv);
    return dist.distance(svtx, pv);
  }

  // dot product between SV and PV (adapted from DeepNTuples)
  float vertexDdotP(const reco::VertexCompositePtrCandidate &sv, const reco::Vertex &pv)  {
    reco::Candidate::Vector p = sv.momentum();
    reco::Candidate::Vector d(sv.vx() - pv.x(), sv.vy() - pv.y(), sv.vz() - pv.z());
    return p.Unit().Dot(d.Unit());
  }

  // helper to order vertices by significance (adapted from DeepNTuples)
  template < typename SVType, typename PVType>
  bool sv_vertex_comparator(const SVType & sva, const SVType & svb, const PVType & pv) {
    auto adxy = vertexDxy(sva,pv); 
    auto bdxy = vertexDxy(svb,pv); 
    float aval= adxy.value();
    float bval= bdxy.value();
    float aerr= adxy.error();
    float berr= bdxy.error();

    float asig= catch_infs(aval/aerr,0.);
    float bsig= catch_infs(bval/berr,0.);
    return bsig<asig;
  }

  // write tagging variables to vector (adapted from DeepNTuples)
  template <typename T>
  int dump_vector(reco::TaggingVariableList& from, T* to,
                  reco::btau::TaggingVariableName name, const size_t max) {
    std::vector<T> vals = from.getList(name ,false);
    size_t size=std::min(vals.size(),max);
    if(size > 0){
      for(size_t i=0;i<vals.size();i++){
        to[i]=catch_infs(vals[i],-0.1);
      }
    }
    return size;
  }

  // compute minimum dr between SVs and a candidate (from DeepNTuples, now polymorphic)
  float mindrsvpfcand(const std::vector<reco::VertexCompositePtrCandidate> & svs, 
                      const reco::Candidate* cand, float mindr=0.4) {

    for (unsigned int i0=0; i0<svs.size(); ++i0) {

        float tempdr = reco::deltaR(svs[i0],*cand);
        if (tempdr<mindr) { mindr = tempdr; }

    }
    return mindr;
  }



}

#endif //RecoBTag_DeepFlavour_deep_helpers_h
