#ifndef __DataFormats_PatCandidates_StoppedTrack_h__
#define __DataFormats_PatCandidates_StoppedTrack_h__

/*
  \class    pat::StoppedTrack StoppedTrack.h "DataFormats/PatCandidates/interface/StoppedTrack.h"
  \brief Small class to store key info on stopped tracks
   pat::StoppedTrack stores important info on stopped tracks. Draws from
   packedPFCandidates, lostTracks, and generalTracks.
  \author   Bennett Marsh
*/

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "PhysicsTools/PatUtils/interface/MiniIsolation.h"

namespace pat {
    class StoppedTrack;
    typedef std::vector<StoppedTrack>  StoppedTrackCollection; 
}


namespace pat {

    class StoppedTrack : public reco::LeafCandidate {

      public:
        typedef MiniIsolation Isolation;

        StoppedTrack() :
          LeafCandidate(0, LorentzVector(0,0,0,0)),
          isolationDR03_({0.,0.,0.,0.}),
          miniIsolation_(pat::MiniIsolation({0,0,0,0})),
          dz_(0.), dxy_(0.), dzError_(0.), dxyError_(0.), trackQuality_(0),
          dEdxStrip_(0), dEdxPixel_(0), hitPattern_(reco::HitPattern()),
          packedCandRef_(PackedCandidateRef()) {}

        explicit StoppedTrack(Isolation iso, MiniIsolation miniiso, 
                              LorentzVector p4, int charge, int id, 
                              float dz, float dxy, float dzError, float dxyError,
                              reco::HitPattern hp, float dEdxS, float dEdxP, int tkQual,
                              PackedCandidateRef pcref) :
          LeafCandidate(charge, p4, Point(0.,0.,0.), id),
          isolationDR03_(iso),
          miniIsolation_(miniiso),
          dz_(dz), dxy_(dxy), dzError_(dzError), dxyError_(dxyError),
          trackQuality_(tkQual), dEdxStrip_(dEdxS), dEdxPixel_(dEdxP), 
          hitPattern_(hp), 
          packedCandRef_(pcref) {}

        ~StoppedTrack() {}

        Isolation isolationDR03() const  { return isolationDR03_; }
        float chargedHadronIso() const   { return isolationDR03_.chiso; }
        float neutralHadronIso() const   { return isolationDR03_.nhiso; }
        float photonIso() const          { return isolationDR03_.phiso; }
        float puChargedHadronIso() const { return isolationDR03_.puiso; }
        void setIsolationDR03(float ch, float nh, float ph, float pu){ isolationDR03_={ch, nh, ph, pu}; }

        MiniIsolation miniPFIsolation() const { return miniIsolation_; }
        float chargedHadronMiniIso() const    { return miniIsolation_.chiso; }
        float neutralHadronMiniIso() const    { return miniIsolation_.nhiso; }
        float photonMiniIso() const           { return miniIsolation_.phiso; }
        float puChargedHadronMiniIso() const  { return miniIsolation_.puiso; }
        void setMiniPFIsolation(MiniIsolation iso){ miniIsolation_ = iso; }

        float dz() const { return dz_; }
        void setDz(float dz){ dz_=dz; }
        float dzError() const { return dzError_; }
        void setDzError(float dzError){ dzError_=dzError; }

        float dxy() const { return dxy_; }
        void setDxy(float dxy){ dxy_=dxy; }        
        float dxyError() const { return dxyError_; }
        void setDxyError(float dxyError){ dxyError_=dxyError; }

        bool isHighPurityTrack() const 
          {  return (trackQuality_ & (1 << reco::TrackBase::highPurity)) >> reco::TrackBase::highPurity; }
        bool isTightTrack() const 
          {  return (trackQuality_ & (1 << reco::TrackBase::tight)) >> reco::TrackBase::tight; }
        bool isLooseTrack() const 
          {  return (trackQuality_ & (1 << reco::TrackBase::loose)) >> reco::TrackBase::loose; }
        void setTrackQuality(int tq) { trackQuality_ = tq; }

        reco::HitPattern hitPattern() const { return hitPattern_; }
        void setHitPattern(reco::HitPattern hp) { hitPattern_ = hp; }
        
        float dEdxStrip() const { return dEdxStrip_; }
        void setDeDxStrip(float dEdxStrip) { dEdxStrip_ = dEdxStrip; }
        float dEdxPixel() const { return dEdxPixel_; }
        void setDeDxPixel(float dEdxPixel) { dEdxPixel_ = dEdxPixel; }

        PackedCandidateRef packedCandRef() const { return packedCandRef_; }
        void setPackedCandRef(PackedCandidateRef ref){ packedCandRef_ = ref; }

      protected:
        Isolation isolationDR03_;
        MiniIsolation miniIsolation_;
        float dz_, dxy_, dzError_, dxyError_;        
        int trackQuality_;
        float dEdxStrip_, dEdxPixel_; //in MeV/mm

        reco::HitPattern hitPattern_;

        PackedCandidateRef packedCandRef_;

    };

}


#endif
