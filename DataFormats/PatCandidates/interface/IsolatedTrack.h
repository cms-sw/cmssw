#ifndef __DataFormats_PatCandidates_IsolatedTrack_h__
#define __DataFormats_PatCandidates_IsolatedTrack_h__

/*
  \class    pat::IsolatedTrack IsolatedTrack.h "DataFormats/PatCandidates/interface/IsolatedTrack.h"
  \brief Small class to store key info isolated packed cands    
   pat::IsolatedTrack stores important info on isolated packed PF candidates,
   as a convenience to analyses to prevent having to loop over all pfcands.
  \author   Bennett Marsh
*/

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "PhysicsTools/PatUtils/interface/MiniIsolation.h"

namespace pat {
    class IsolatedTrack;
    typedef std::vector<IsolatedTrack>  IsolatedTrackCollection; 
    typedef edm::Ptr<pat::PackedCandidate> PackedCandidatePtr;
}


namespace pat {

    class IsolatedTrack : public reco::LeafCandidate {

      public:
        typedef MiniIsolation Isolation;

        IsolatedTrack() :
          LeafCandidate(0, LorentzVector(0,0,0,0)),
          isolationDR03_({0.,0.,0.,0.}),
          miniIsolation_(pat::MiniIsolation({0,0,0,0})),
          dz_(0.), dxy_(0.),
          refToCand_(PackedCandidatePtr()) {}

        explicit IsolatedTrack(Isolation iso, MiniIsolation miniiso, 
                               LorentzVector p4, int charge, int id, 
                               float dz, float dxy, PackedCandidatePtr ref) :
          LeafCandidate(charge, p4, Point(0.,0.,0.), id),
          isolationDR03_(iso),
          miniIsolation_(miniiso),
          dz_(dz), dxy_(dxy),
          refToCand_(ref) {}

        ~IsolatedTrack() {}

        Isolation isolationDR03(){ return isolationDR03_; }
        void setIsolationDR03(float ch, float nh, float ph, float pu){ isolationDR03_={ch, nh, ph, pu}; }

        MiniIsolation miniPFIsolation(){ return miniIsolation_; }
        void setMiniPFIsolation(MiniIsolation iso){ miniIsolation_ = iso; }

        float dz(){ return dz_; }
        void setDz(float dz){ dz_=dz; }

        float dxy(){ return dxy_; }
        void setDxy(float dxy){ dxy_=dxy; }
        
        PackedCandidatePtr refToCand(){ return refToCand_; }
        void setPackedCandRef(PackedCandidatePtr ref){ refToCand_ = ref; }

      protected:
        Isolation isolationDR03_;
        MiniIsolation miniIsolation_;
        float dz_, dxy_;

        PackedCandidatePtr refToCand_;
    };

}


#endif
