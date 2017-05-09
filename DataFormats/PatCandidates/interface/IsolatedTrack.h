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
#include "DataFormats/PatCandidates/interface/PFIsolation.h"

namespace pat {

    class IsolatedTrack : public reco::LeafCandidate {

      public:

        IsolatedTrack() :
          LeafCandidate(0, LorentzVector(0,0,0,0)),
          pfIsolationDR03_(pat::PFIsolation()),
          miniIsolation_(pat::PFIsolation()),
          dz_(0.), dxy_(0.),
          packedCandRef_(PackedCandidateRef()) {}

        explicit IsolatedTrack(const PFIsolation &iso, const PFIsolation &miniiso,
                               const LorentzVector &p4, int charge, int id,
                               float dz, float dxy, const PackedCandidateRef &ref) :
          LeafCandidate(charge, p4, Point(0.,0.,0.), id),
          pfIsolationDR03_(iso),
          miniIsolation_(miniiso),
          dz_(dz), dxy_(dxy),
          packedCandRef_(ref) {}

        ~IsolatedTrack() {}

        const PFIsolation& pfIsolationDR03() const  { return pfIsolationDR03_; }
        const PFIsolation& miniPFIsolation() const { return miniIsolation_; }
        float dz() const { return dz_; }
        float dxy() const { return dxy_; }
        
        const PackedCandidateRef& packedCandRef() const { return packedCandRef_; }

      protected:
        PFIsolation pfIsolationDR03_;
        PFIsolation miniIsolation_;
        float dz_, dxy_;

        PackedCandidateRef packedCandRef_;
    };


    typedef std::vector<IsolatedTrack>  IsolatedTrackCollection;
}


#endif
