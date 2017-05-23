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
          matchedCaloJetEmEnergy_(0.), matchedCaloJetHadEnergy_(0.),
          dz_(0.), dxy_(0.), dzError_(0.), dxyError_(0.), fromPV_(-1), trackQuality_(0),
          dEdxStrip_(0), dEdxPixel_(0), hitPattern_(reco::HitPattern()),
          crossedEcalIds_(std::vector<DetId>()), crossedHcalIds_(std::vector<HcalDetId>()),
          crossedEcalStatus_(std::vector<EcalChannelStatusCode>()),
          crossedHcalStatus_(std::vector<uint32_t>()),
          packedCandRef_(PackedCandidateRef()) {}

        explicit StoppedTrack(const PFIsolation &iso, const PFIsolation &miniiso, float caloJetEm, float caloJetHad,
                              const LorentzVector &p4, int charge, int id,
                              float dz, float dxy, float dzError, float dxyError,
                              const reco::HitPattern &hp, float dEdxS, float dEdxP, int fromPV, int tkQual,
                              const std::vector<DetId> & ecalid, const std::vector<HcalDetId> & hcalid,
                              const std::vector<EcalChannelStatusCode> &ecalst,
                              const std::vector<uint32_t> & hcalst,
                              const PackedCandidateRef &pcref) :
          LeafCandidate(charge, p4, Point(0.,0.,0.), id),
          pfIsolationDR03_(iso), miniIsolation_(miniiso), 
          matchedCaloJetEmEnergy_(caloJetEm), matchedCaloJetHadEnergy_(caloJetHad),
          dz_(dz), dxy_(dxy), dzError_(dzError), dxyError_(dxyError),
          fromPV_(fromPV), trackQuality_(tkQual), dEdxStrip_(dEdxS), dEdxPixel_(dEdxP), 
          hitPattern_(hp), crossedEcalIds_(ecalid), crossedHcalIds_(hcalid),
          crossedEcalStatus_(ecalst), crossedHcalStatus_(hcalst),
          packedCandRef_(pcref) {}

        ~StoppedTrack() {}

        const PFIsolation& pfIsolationDR03() const  { return pfIsolationDR03_; }

        const PFIsolation& miniPFIsolation() const { return miniIsolation_; }

        float matchedCaloJetEmEnergy() const { return matchedCaloJetEmEnergy_; }
        float matchedCaloJetHadEnergy() const { return matchedCaloJetHadEnergy_; }

        float dz() const { return dz_; }
        float dzError() const { return dzError_; }
        float dxy() const { return dxy_; }
        float dxyError() const { return dxyError_; }

        int fromPV() const { return fromPV_; }

        bool isHighPurityTrack() const 
          {  return (trackQuality_ & (1 << reco::TrackBase::highPurity)) >> reco::TrackBase::highPurity; }
        bool isTightTrack() const 
          {  return (trackQuality_ & (1 << reco::TrackBase::tight)) >> reco::TrackBase::tight; }
        bool isLooseTrack() const 
          {  return (trackQuality_ & (1 << reco::TrackBase::loose)) >> reco::TrackBase::loose; }

        const reco::HitPattern& hitPattern() const { return hitPattern_; }
        
        float dEdxStrip() const { return dEdxStrip_; }
        float dEdxPixel() const { return dEdxPixel_; }

        const std::vector<DetId>&     crossedEcalIds() const { return crossedEcalIds_; }
        const std::vector<HcalDetId>& crossedHcalIds() const { return crossedHcalIds_; }

        const std::vector<EcalChannelStatusCode>& crossedEcalStatus() const { return crossedEcalStatus_; }
        // HcalChannelStatus duplicates DetId info.
        // Only store the status code value, and here construct the full object
        std::vector<HcalChannelStatus> crossedHcalStatus() const {
            std::vector<HcalChannelStatus> status;
            for(unsigned int i=0; i<crossedHcalIds_.size(); i++)
                status.push_back(HcalChannelStatus(crossedHcalIds_[i].rawId(), crossedHcalStatus_[i]));
            return status;
        }

        const PackedCandidateRef& packedCandRef() const { return packedCandRef_; }

      protected:
        PFIsolation pfIsolationDR03_;
        PFIsolation miniIsolation_;
        float matchedCaloJetEmEnergy_;  //energy of nearest calojet within a given dR;
        float matchedCaloJetHadEnergy_;
        float dz_, dxy_, dzError_, dxyError_;        
        int fromPV_;  //only stored for packedPFCandidates
        int trackQuality_;
        float dEdxStrip_, dEdxPixel_; //in MeV/mm

        reco::HitPattern hitPattern_;

        std::vector<DetId> crossedEcalIds_;
        std::vector<HcalDetId> crossedHcalIds_;
        std::vector<EcalChannelStatusCode> crossedEcalStatus_;
        std::vector<uint32_t> crossedHcalStatus_;

        PackedCandidateRef packedCandRef_;

    };

    typedef std::vector<StoppedTrack> StoppedTrackCollection;

}

#endif
