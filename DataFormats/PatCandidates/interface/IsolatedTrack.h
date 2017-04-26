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
#include "PhysicsTools/PatUtils/interface/MiniIsolation.h"

namespace pat {
    class IsolatedTrack;
    typedef std::vector<IsolatedTrack>  IsolatedTrackCollection; 
    typedef edm::Ptr<pat::PackedCandidate> PackedCandidatePtr;
}


namespace pat {

    class IsolatedTrack {

      public:
        typedef math::XYZTLorentzVector LorentzVector;

        IsolatedTrack() :
          trackIsoDR03_(0.),
          miniIso_(pat::MiniIsolation({0,0,0,0})),
          p4_( LorentzVector(0.,0.,0.,0.)),
          pdgId_(0),
          refToCand_(PackedCandidatePtr()) {}

        explicit IsolatedTrack(float tkiso, MiniIsolation miniiso, LorentzVector p4, int id, PackedCandidatePtr ref) :
          trackIsoDR03_(tkiso),
          miniIso_(miniiso),
          p4_(p4),
          pdgId_(id),
          refToCand_(ref) {}

        ~IsolatedTrack() {}

        float trackIsoDR03(){ return trackIsoDR03_; }
        void setTrackIsoDR03(float iso){ trackIsoDR03_ = iso; }

        MiniIsolation miniPFIsolation(){ return miniIso_; }
        void setMiniPFIsolation(MiniIsolation iso){ miniIso_ = iso; }

        LorentzVector p4(){ return p4_; }
        void setP4(LorentzVector p4){ p4_ = p4; }

        int pdgId(){ return pdgId_; }
        void setPdgId(int id){ pdgId_ = id; }

        PackedCandidatePtr refToCand(){ return refToCand_; }
        void setPackedCandRef(PackedCandidatePtr ref){ refToCand_ = ref; }

      protected:
        float trackIsoDR03_;
        MiniIsolation miniIso_;
        LorentzVector p4_;
        int pdgId_;

        PackedCandidatePtr refToCand_;
    };

}


#endif
