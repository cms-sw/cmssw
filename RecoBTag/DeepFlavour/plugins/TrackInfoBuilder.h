#ifndef RecoBTag_DeepFlavour_TrackInfoBuilder_h
#define RecoBTag_DeepFlavour_TrackInfoBuilder_h

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "TVector3.h"

namespace btagbtvdeep{

// adapted from DeepNtuples
class TrackInfoBuilder{
public:
    TrackInfoBuilder(edm::ESHandle<TransientTrackBuilder> & build):
        builder_(build),
        trackMomentum_(0),
        trackEta_(0),
        trackEtaRel_(0),
        trackPtRel_(0),
        trackPPar_(0),
        trackDeltaR_(0),
        trackPtRatio_(0),
        trackPParRatio_(0),
        trackSip2dVal_(0),
        trackSip2dSig_(0),
        trackSip3dVal_(0),
        trackSip3dSig_(0),

        trackJetDistVal_(0),
        trackJetDistSig_(0)
{


}

    void buildTrackInfo(const reco::Candidate * candidate ,const math::XYZVector&  jetDir, GlobalVector refjetdirection, const reco::Vertex & pv){
        TVector3 jetDir3(jetDir.x(),jetDir.y(),jetDir.z());


        // deal with PAT/AOD polymorphism to get track
        const reco::Track * track_ptr = nullptr;
        auto packed_candidate = dynamic_cast<const pat::PackedCandidate *>(candidate);
        auto pf_candidate = dynamic_cast<const reco::PFCandidate *>(candidate);
        if (pf_candidate) {
          track_ptr = pf_candidate->bestTrack(); // trackRef was sometimes null
        } else if (packed_candidate && packed_candidate->hasTrackDetails()) {
          // if PackedCandidate does not have TrackDetails this gives an Exception
          // because unpackCovariance might be called for pseudoTrack/bestTrack
          track_ptr = &(packed_candidate->pseudoTrack());
        }

        if(!track_ptr) {
          TVector3 trackMom3(
            candidate->momentum().x(),
            candidate->momentum().y(),
            candidate->momentum().z()
            );
          trackMomentum_=candidate->p();
          trackEta_= candidate->eta();
          trackEtaRel_=reco::btau::etaRel(jetDir, candidate->momentum());
          trackPtRel_=trackMom3.Perp(jetDir3);
          trackPPar_=jetDir.Dot(candidate->momentum());
          trackDeltaR_=reco::deltaR(candidate->momentum(), jetDir);
          trackPtRatio_=trackMom3.Perp(jetDir3) / candidate->p();
          trackPParRatio_=jetDir.Dot(candidate->momentum()) / candidate->p();
          trackSip2dVal_=0.;
          trackSip2dSig_=0.;
          trackSip3dVal_=0.;
          trackSip3dSig_=0.;
          trackJetDistVal_=0.;
          trackJetDistSig_=0.;
          return;
        }

        math::XYZVector trackMom = track_ptr->momentum();
        double trackMag = std::sqrt(trackMom.Mag2());
        TVector3 trackMom3(trackMom.x(),trackMom.y(),trackMom.z());

        trackMomentum_=std::sqrt(trackMom.Mag2());
        trackEta_= trackMom.Eta();
        trackEtaRel_=reco::btau::etaRel(jetDir, trackMom);
        trackPtRel_=trackMom3.Perp(jetDir3);
        trackPPar_=jetDir.Dot(trackMom);
        trackDeltaR_=reco::deltaR(trackMom, jetDir);
        trackPtRatio_=trackMom3.Perp(jetDir3) / trackMag;
        trackPParRatio_=jetDir.Dot(trackMom) / trackMag;

        reco::TransientTrack transientTrack;
        transientTrack=builder_->build(*track_ptr);
        Measurement1D meas_ip2d=IPTools::signedTransverseImpactParameter(transientTrack, refjetdirection, pv).second;
        Measurement1D meas_ip3d=IPTools::signedImpactParameter3D(transientTrack, refjetdirection, pv).second;
        Measurement1D jetdist=IPTools::jetTrackDistance(transientTrack, refjetdirection, pv).second;
        trackSip2dVal_=static_cast<float>(meas_ip2d.value());
        trackSip2dSig_=static_cast<float>(meas_ip2d.significance());
        trackSip3dVal_=static_cast<float>(meas_ip3d.value());
        trackSip3dSig_=static_cast<float>(meas_ip3d.significance());
        trackJetDistVal_=static_cast<float>(jetdist.value());
        trackJetDistSig_=static_cast<float>(jetdist.significance());

    }

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

#endif //RecoBTag_DeepFlavour_TrackInfoBuilder_h
