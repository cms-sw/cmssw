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
#include "DataFormats/PatCandidates/interface/PFIsolation.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"


namespace pat {

    class StoppedTrack : public reco::LeafCandidate {

      public:

        StoppedTrack() :
          LeafCandidate(0, LorentzVector(0,0,0,0)),
          pfIsolationDR03_(pat::PFIsolation()),
          miniIsolation_(pat::PFIsolation()),
          dz_(0.), dxy_(0.), dzError_(0.), dxyError_(0.), trackQuality_(0),
          dEdxStrip_(0), dEdxPixel_(0), hitPattern_(reco::HitPattern()),
          crossedEcalIds_(std::vector<DetId>()), crossedHcalIds_(std::vector<HcalDetId>()),
          crossedEcalStatus_(std::vector<EcalChannelStatusCode>()),
          crossedHcalStatus_(std::vector<HcalChannelStatus>()), 
          packedCandRef_(PackedCandidateRef()) {}

        explicit StoppedTrack(const PFIsolation &iso, const PFIsolation &miniiso,
                              const LorentzVector &p4, int charge, int id,
                              float dz, float dxy, float dzError, float dxyError,
                              const reco::HitPattern &hp, float dEdxS, float dEdxP, int tkQual,
                              const std::vector<DetId> & ecalid, const std::vector<HcalDetId> & hcalid,
                              const std::vector<EcalChannelStatusCode> &ecalst,
                              const std::vector<HcalChannelStatus> & hcalst, 
                              const PackedCandidateRef &pcref) :
          LeafCandidate(charge, p4, Point(0.,0.,0.), id),
          pfIsolationDR03_(iso),
          miniIsolation_(miniiso),
          dz_(dz), dxy_(dxy), dzError_(dzError), dxyError_(dxyError),
          trackQuality_(tkQual), dEdxStrip_(dEdxS), dEdxPixel_(dEdxP), 
          hitPattern_(hp), crossedEcalIds_(ecalid), crossedHcalIds_(hcalid),
          crossedEcalStatus_(ecalst), crossedHcalStatus_(hcalst),
          packedCandRef_(pcref) {}

        ~StoppedTrack() {}

        const PFIsolation& pfIsolationDR03() const  { return pfIsolationDR03_; }

        const PFIsolation& miniPFIsolation() const { return miniIsolation_; }

        float dz() const { return dz_; }
        float dzError() const { return dzError_; }
        float dxy() const { return dxy_; }
        float dxyError() const { return dxyError_; }

        bool isHighPurityTrack() const 
          {  return (trackQuality_ & (1 << reco::TrackBase::highPurity)) >> reco::TrackBase::highPurity; }
        bool isTightTrack() const 
          {  return (trackQuality_ & (1 << reco::TrackBase::tight)) >> reco::TrackBase::tight; }
        bool isLooseTrack() const 
          {  return (trackQuality_ & (1 << reco::TrackBase::loose)) >> reco::TrackBase::loose; }

        const reco::HitPattern& hitPattern() const { return hitPattern_; }
        
        float dEdxStrip() const { return dEdxStrip_; }
        float dEdxPixel() const { return dEdxPixel_; }

        const std::vector<DetId>& crossedEcalIds() const { return crossedEcalIds_; }
        const std::vector<HcalDetId>& crossedHcalIds() const { return crossedHcalIds_; }

        const std::vector<EcalChannelStatusCode>& crossedEcalStatus() const { return crossedEcalStatus_; }
        const std::vector<HcalChannelStatus>& crossedHcalStatus() const { return crossedHcalStatus_; }

        const PackedCandidateRef& packedCandRef() const { return packedCandRef_; }

      protected:
        PFIsolation pfIsolationDR03_;
        PFIsolation miniIsolation_;
        float dz_, dxy_, dzError_, dxyError_;        
        int trackQuality_;
        float dEdxStrip_, dEdxPixel_; //in MeV/mm

        reco::HitPattern hitPattern_;

        std::vector<DetId> crossedEcalIds_;
        std::vector<HcalDetId> crossedHcalIds_;
        std::vector<EcalChannelStatusCode> crossedEcalStatus_;
        std::vector<HcalChannelStatus> crossedHcalStatus_;

        PackedCandidateRef packedCandRef_;

    };

}

namespace pat {
    typedef std::vector<StoppedTrack>  StoppedTrackCollection;
    typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > LorentzVector;
}


#endif
