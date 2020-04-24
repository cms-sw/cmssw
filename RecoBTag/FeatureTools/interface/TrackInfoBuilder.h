#ifndef RecoBTag_FeatureTools_TrackInfoBuilder_h
#define RecoBTag_FeatureTools_TrackInfoBuilder_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

namespace btagbtvdeep{

// adapted from DeepNtuples
class TrackInfoBuilder{
public:
    TrackInfoBuilder(edm::ESHandle<TransientTrackBuilder> & build);

    void buildTrackInfo(const reco::Candidate * candidate ,const math::XYZVector&  jetDir, GlobalVector refjetdirection, const reco::Vertex & pv);
    const float getTrackDeltaR() const {return trackDeltaR_;}
    const float getTrackEta() const {return trackEta_;}
    const float getTrackEtaRel() const {return trackEtaRel_;}
    const float getTrackJetDistSig() const {return trackJetDistSig_;}
    const float getTrackJetDistVal() const {return trackJetDistVal_;}
    const float getTrackMomentum() const {return trackMomentum_;}
    const float getTrackPPar() const {return trackPPar_;}
    const float getTrackPParRatio() const {return trackPParRatio_;}
    const float getTrackPtRatio() const {return trackPtRatio_;}
    const float getTrackPtRel() const {return trackPtRel_;}
    const float getTrackSip2dSig() const {return trackSip2dSig_;}
    const float getTrackSip2dVal() const {return trackSip2dVal_;}
    const float getTrackSip3dSig() const {return trackSip3dSig_;}
    const float getTrackSip3dVal() const {return trackSip3dVal_;}

private:

    edm::ESHandle<TransientTrackBuilder> builder_;

    float trackMomentum_;
    float trackEta_;
    float trackEtaRel_;
    float trackPtRel_;
    float trackPPar_;
    float trackDeltaR_;
    float trackPtRatio_;
    float trackPParRatio_;
    float trackSip2dVal_;
    float trackSip2dSig_;
    float trackSip3dVal_;
    float trackSip3dSig_;

    float trackJetDistVal_;
    float trackJetDistSig_;

};

}

#endif //RecoBTag_FeatureTools_TrackInfoBuilder_h
