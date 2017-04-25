#ifndef __DataFormats_PatCandidates_IsolatedTrack_h__
#define __DataFormats_PatCandidates_IsolatedTrack_h__

/*
  \class    pat::IsolatedTrack IsolatedTrack.h "DataFormats/PatCandidates/interface/IsolatedTrack.h"
  \brief Small class to store key info isolated packed cands    
   pat::IsolatedTrack stores important info on isolated packed PF candidates,
   as a convenience to analyses to prevent having to loop over all pfcands.
  \author   Bennett Marsh
*/


#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

namespace pat {
    class IsolatedTrack;
    typedef std::vector<IsolatedTrack>  IsolatedTrackCollection; 
}


namespace pat {

    class IsolatedTrack {

      public:
        typedef math::XYZTLorentzVector LorentzVector;

        IsolatedTrack() :
          trackIsoDR03_(0.),
          miniTrackIso_(0.),
          p4_( LorentzVector(0.,0.,0.,0.)),
          pdgId_(0),
          /* refToCand_(reco::CandidatePtr()) {} */
          candIndex_(-1) {}

        /* explicit IsolatedTrack(float tkiso, float miniiso, LorentzVector p4, int id, reco::CandidatePtr ref) : */
        explicit IsolatedTrack(float tkiso, float miniiso, LorentzVector p4, int id, int idx) :
          trackIsoDR03_(tkiso),
          miniTrackIso_(miniiso),
          p4_(p4),
          pdgId_(id),
          candIndex_(idx) {}

        ~IsolatedTrack() {}

        float trackIsoDR03(){ return trackIsoDR03_; }
        void setTrackIsoDR03(float iso){ trackIsoDR03_ = iso; }

        float miniTrackIso(){ return miniTrackIso_; }
        void setMiniTrackIso(float iso){ miniTrackIso_ = iso; }

        LorentzVector p4(){ return p4_; }
        void setP4(LorentzVector p4){ p4_ = p4; }

        int pdgId(){ return pdgId_; }
        void setPdgId(int id){ pdgId_ = id; }

        /* reco::CandidatePtr refToCand(){ return refToCand_; } */
        /* void setPackedCandRef(reco::CandidatePtr ref){ refToCand_ = ref; } */

        int candIndex(){ return candIndex_; }
        void setCandIndex(int idx){ candIndex_ = idx; }

      protected:
        float trackIsoDR03_;
        float miniTrackIso_;
        LorentzVector p4_;
        int pdgId_;

        /* reco::CandidatePtr refToCand_; */
        int candIndex_; //index of corresponding index in PackedPFCandidates
    };

}


#endif
