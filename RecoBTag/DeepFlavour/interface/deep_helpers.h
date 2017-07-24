#ifndef RecoBTag_DeepFlavour_deep_helpers_h
#define RecoBTag_DeepFlavour_deep_helpers_h

#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/IPTools/interface/IPTools.h"



namespace deep {

  // adapted from DeepNTuples
  inline const float& catch_infs(const float& in,
                                 const float& replace_value) {
    if(in==in){ // why is this ?
      if(std::isinf(in))
        return replace_value;
      else if(in < -1e32 || in > 1e32)
        return replace_value;
      return in;
    }
    return replace_value;
  } 

  // adapted from DeepNTuples
  inline const float catch_infs_and_bound(const float& in,
                                          const float& replace_value,
                                          const float& lowerbound,
                                          const float& upperbound,
                                          const float offset=0.,
                                          const bool& use_offsets = true){
        float withoutinfs=catch_infs(in,replace_value);
        if(withoutinfs+offset<lowerbound) return lowerbound;
        if(withoutinfs+offset>upperbound) return upperbound;
        if(use_offsets)
            withoutinfs+=offset;
        return withoutinfs;
  }


  // adapted from DeepNTuples 
  Measurement1D vertexDxy(const reco::VertexCompositePtrCandidate &svcand, const reco::Vertex &pv)  {
    VertexDistanceXY dist;
    reco::Vertex::CovarianceMatrix csv; svcand.fillVertexCovariance(csv);
    reco::Vertex svtx(svcand.vertex(), csv);
    return dist.distance(svtx, pv);
  }

  // adapted from DeepNTuples 
  Measurement1D vertexD3d(const reco::VertexCompositePtrCandidate &svcand, const reco::Vertex &pv)  {
    VertexDistance3D dist;
    reco::Vertex::CovarianceMatrix csv; svcand.fillVertexCovariance(csv);
    reco::Vertex svtx(svcand.vertex(), csv);
    return dist.distance(svtx, pv);
  }

  // adapted from DeepNTuples 
  float vertexDdotP(const reco::VertexCompositePtrCandidate &sv, const reco::Vertex &pv)  {
    reco::Candidate::Vector p = sv.momentum();
    reco::Candidate::Vector d(sv.vx() - pv.x(), sv.vy() - pv.y(), sv.vz() - pv.z());
    return p.Unit().Dot(d.Unit());
  }

  // adapted from DeepNTuples 
  template < typename SVType, typename PVType>
  bool sv_vertex_comparator(const SVType & sva, const SVType & svb, const PVType & pv) {
    float adxy= vertexDxy(sva,pv).value();
    float bdxy= vertexDxy(svb,pv).value();
    float aerr= vertexDxy(sva,pv).error();
    float berr= vertexDxy(svb,pv).error();

    float asig= catch_infs(adxy/aerr,0.);
    float bsig= catch_infs(bdxy/berr,0.);
    return bsig<asig;
  }

  // adapted from DeepNTuples 
  template <typename T>
  int dump_vector(reco::TaggingVariableList& from, T* to,
                  reco::btau::TaggingVariableName name, const size_t& max) {
    std::vector<T> vals = from.getList(name ,false);
    size_t size=std::min(vals.size(),max);
    if(size > 0){
      for(size_t i=0;i<vals.size();i++){
        to[i]=catch_infs(vals.at(i),-0.1);
      }
    }
    return size;
  }

  // adapted from DeepNtuples (now polymorphic)
  float mindrsvpfcand(const std::vector<reco::VertexCompositePtrCandidate> svs, 
                      const reco::Candidate* cand, float mindr=0.4) {

    for (unsigned int i0=0; i0<svs.size(); ++i0) {

        float tempdr = reco::deltaR(svs[i0],*cand);
        if (tempdr<mindr) { mindr = tempdr; }

    }
    return mindr;
  }



}

#endif //RecoBTag_DeepFlavour_deep_helpers_h
