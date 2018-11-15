#include "RecoBTag/FeatureTools/interface/deep_helpers.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

namespace btagbtvdeep {

  constexpr static int qualityMap[8]  = {1,0,1,1,4,4,5,6};

  enum qualityFlagsShiftsAndMasks {
    assignmentQualityMask = 0x7,
    assignmentQualityShift = 0,
    trackHighPurityMask  = 0x8,
    trackHighPurityShift=3,
    lostInnerHitsMask = 0x30,
    lostInnerHitsShift=4,
    muonFlagsMask = 0x0600,
    muonFlagsShift=9
  };


  // remove infs and NaNs with value  (adapted from DeepNTuples)
  const float catch_infs(const float in,
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
  const float catch_infs_and_bound(const float in,
				   const float replace_value,
				   const float lowerbound,
				   const float upperbound,
				   const float offset,
				   const bool use_offsets){
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

  // compute minimum dr between SVs and a candidate (from DeepNTuples, now polymorphic)
  float mindrsvpfcand(const std::vector<reco::VertexCompositePtrCandidate> & svs,
                      const reco::Candidate* cand, float mindr) {

    for (unsigned int i0=0; i0<svs.size(); ++i0) {

      float tempdr = reco::deltaR(svs[i0],*cand);
      if (tempdr<mindr) { mindr = tempdr; }

    }
    return mindr;
  }

  // instantiate template
  template bool sv_vertex_comparator<reco::VertexCompositePtrCandidate, reco::Vertex>(const reco::VertexCompositePtrCandidate&,
										      const reco::VertexCompositePtrCandidate&,
										      const reco::Vertex&);

  float vtx_ass_from_pfcand(const reco::PFCandidate& pfcand, int pv_ass_quality, const reco::VertexRef & pv) {
    float vtx_ass = pat::PackedCandidate::PVAssociationQuality(qualityMap[pv_ass_quality]);
    if (pfcand.trackRef().isNonnull() &&
        pv->trackWeight(pfcand.trackRef()) > 0.5 &&
        pv_ass_quality == 7) {
      vtx_ass = pat::PackedCandidate::UsedInFitTight;
    }
    return vtx_ass;
  }

  float quality_from_pfcand(const reco::PFCandidate& pfcand) {
    const auto & pseudo_track =  (pfcand.bestTrack()) ? *pfcand.bestTrack() : reco::Track();
    // conditions from PackedCandidate producer
    bool highPurity = pfcand.trackRef().isNonnull() && pseudo_track.quality(reco::Track::highPurity);
    // do same bit operations than in PackedCandidate
    uint16_t qualityFlags = 0;
    qualityFlags = (qualityFlags & ~trackHighPurityMask) | ((highPurity << trackHighPurityShift) & trackHighPurityMask);
    bool isHighPurity = (qualityFlags & trackHighPurityMask)>>trackHighPurityShift;
    // to do as in TrackBase
    uint8_t quality = (1 << reco::TrackBase::loose);
    if (isHighPurity) {
      quality |= (1 << reco::TrackBase::highPurity);
    }
    return quality;
  }

  float lost_inner_hits_from_pfcand(const reco::PFCandidate& pfcand) {
    const auto & pseudo_track =  (pfcand.bestTrack()) ? *pfcand.bestTrack() : reco::Track();
    // conditions from PackedCandidate producer
    bool highPurity = pfcand.trackRef().isNonnull() && pseudo_track.quality(reco::Track::highPurity);
    // do same bit operations than in PackedCandidate
    uint16_t qualityFlags = 0;
    qualityFlags = (qualityFlags & ~trackHighPurityMask) | ((highPurity << trackHighPurityShift) & trackHighPurityMask);
    return int16_t((qualityFlags & lostInnerHitsMask)>>lostInnerHitsShift)-1;
  }

}

