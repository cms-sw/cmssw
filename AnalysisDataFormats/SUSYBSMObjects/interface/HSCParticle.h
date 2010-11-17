#ifndef HSCParticle_H
#define HSCParticle_H
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>

#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"

namespace susybsm {

 /// define arbitration schemes
 namespace HSCParticleType{
    enum Type {globalMuon, trackerMuon, matchedStandAloneMuon, standAloneMuon, innerTrack, unknown };
 }
 
 class RPCHit4D
  {
   public:
     int id;
     int bx;
     GlobalPoint gp;
     bool operator<(const RPCHit4D& other) const {
       return gp.mag() < other.gp.mag();
     }
  };

 class RPCBetaMeasurement
  {
   public:
     bool isCandidate;
     std::vector<RPCHit4D> hits;
     float beta;
  };

 class CaloBetaMeasurement
  {
   public:
     float hcalCrossedEnergy, ecalCrossedEnergy, hoCrossedEnergy;
     float ecalDeDx;
     float ecal3by3dir, ecal5by5dir;
     float hcal3by3dir, hcal5by5dir;
     float trkIsoDr;
     float ecalTime, ecalTimeError;
     float ecalBeta, ecalBetaError;
     float ecalInvBetaError;
     int ecalCrysCrossed;
     std::vector<float> ecalSwissCrossKs;
     std::vector<float> ecalE1OverE9s;
     std::vector<float> ecalTrackLengths;
     std::vector<float> ecalEnergies;
     std::vector<float> ecalTimes;
     std::vector<float> ecalTimeErrors;
     std::vector<float> ecalChi2s;
     std::vector<float> ecalOutOfTimeChi2s;
     std::vector<float> ecalOutOfTimeEnergies;
     std::vector<DetId> ecalDetIds;
     std::vector<GlobalPoint> ecalTrackExitPositions;

     CaloBetaMeasurement()
     {
       hcalCrossedEnergy = -9999;
       ecalCrossedEnergy = -9999;
       hoCrossedEnergy = -9999;
       ecalDeDx = -9999;
       ecal3by3dir = -9999;
       ecal5by5dir = -9999;
       hcal3by3dir = -9999;
       hcal5by5dir = -9999;
       trkIsoDr = -9999;
       ecalTime = -9999;
       ecalTimeError = -9999;
       ecalBeta = -9999;
       ecalBetaError = -9999;
       ecalInvBetaError = -9999;
       ecalCrysCrossed = 0;
     }
  };

 class HSCParticle 
  {
   public:
      // constructor
      HSCParticle(){
         hasMuonRef_          = false;
         hasTrackRef_         = false;
         hasDedxEstim1_       = false;
         hasDedxEstim2_       = false;
         hasDedxEstim3_       = false;
         hasDedxEstim4_       = false;
         hasDedxEstim5_       = false;
         hasDedxEstim6_       = false;
         hasDedxDiscrim1_     = false;
         hasDedxDiscrim2_     = false;
         hasDedxDiscrim3_     = false;
         hasDedxDiscrim4_     = false;
         hasDedxDiscrim5_     = false;
         hasDedxDiscrim6_     = false;
         hasMuonTimeDt_       = false;
         hasMuonTimeCsc_      = false;
         hasMuonTimeCombined_ = false;
         hasRpc_              = false;
         hasCalo_             = false;
      }

      // check available infos
      bool  hasMuonRef()          const { return hasMuonRef_;          }
      bool  hasTrackRef()         const { return hasTrackRef_;         }
      bool  hasDedxEstim1()       const { return hasDedxEstim1_;       }
      bool  hasDedxEstim2()       const { return hasDedxEstim2_;       }
      bool  hasDedxEstim3()       const { return hasDedxEstim3_;       }
      bool  hasDedxEstim4()       const { return hasDedxEstim4_;       }
      bool  hasDedxEstim5()       const { return hasDedxEstim5_;       }
      bool  hasDedxEstim6()       const { return hasDedxEstim6_;       }
      bool  hasDedxDiscrim1()     const { return hasDedxDiscrim1_;     }
      bool  hasDedxDiscrim2()     const { return hasDedxDiscrim2_;     }
      bool  hasDedxDiscrim3()     const { return hasDedxDiscrim3_;     }
      bool  hasDedxDiscrim4()     const { return hasDedxDiscrim4_;     }
      bool  hasDedxDiscrim5()     const { return hasDedxDiscrim5_;     }
      bool  hasDedxDiscrim6()     const { return hasDedxDiscrim6_;     }
      bool  hasMuonTimeDt()       const { return hasMuonTimeDt_;       }
      bool  hasMuonTimeCsc()      const { return hasMuonTimeCsc_;      }
      bool  hasMuonTimeCombined() const { return hasMuonTimeCombined_; }
      bool  hasRpcInfo()          const { return hasRpc_;              }
      bool  hasCaloInfo()         const { return hasCalo_;             }

      // set infos
      void setMuon              (const reco::MuonRef&       data) {muonRef_          = data; hasMuonRef_          = true;}
      void setTrack             (const reco::TrackRef&      data) {trackRef_         = data; hasTrackRef_         = true;}
      void setDedxEstimator1    (const reco::DeDxData&      data) {dedxEstim1_       = data; hasDedxEstim1_       = true;}
      void setDedxEstimator2    (const reco::DeDxData&      data) {dedxEstim2_       = data; hasDedxEstim2_       = true;}
      void setDedxEstimator3    (const reco::DeDxData&      data) {dedxEstim3_       = data; hasDedxEstim3_       = true;}
      void setDedxEstimator4    (const reco::DeDxData&      data) {dedxEstim4_       = data; hasDedxEstim4_       = true;}
      void setDedxEstimator5    (const reco::DeDxData&      data) {dedxEstim5_       = data; hasDedxEstim5_       = true;}
      void setDedxEstimator6    (const reco::DeDxData&      data) {dedxEstim6_       = data; hasDedxEstim6_       = true;}
      void setDedxDiscriminator1(const reco::DeDxData&      data) {dedxDiscrim1_     = data; hasDedxDiscrim1_     = true;}
      void setDedxDiscriminator2(const reco::DeDxData&      data) {dedxDiscrim2_     = data; hasDedxDiscrim2_     = true;}
      void setDedxDiscriminator3(const reco::DeDxData&      data) {dedxDiscrim3_     = data; hasDedxDiscrim3_     = true;}
      void setDedxDiscriminator4(const reco::DeDxData&      data) {dedxDiscrim4_     = data; hasDedxDiscrim4_     = true;}
      void setDedxDiscriminator5(const reco::DeDxData&      data) {dedxDiscrim5_     = data; hasDedxDiscrim5_     = true;}
      void setDedxDiscriminator6(const reco::DeDxData&      data) {dedxDiscrim6_     = data; hasDedxDiscrim6_     = true;}
      void setMuonTimeDt        (const reco::MuonTimeExtra& data) {muonTimeDt_       = data; hasMuonTimeDt_       = true;}
      void setMuonTimeCsc       (const reco::MuonTimeExtra& data) {muonTimeCsc_      = data; hasMuonTimeCsc_      = true;}
      void setMuonTimeCombined  (const reco::MuonTimeExtra& data) {muonTimeCombined_ = data; hasMuonTimeCombined_ = true;}
      void setRpc               (const RPCBetaMeasurement&  data) {rpc_              = data; hasRpc_              = true;}
      void setCalo              (const CaloBetaMeasurement& data) {calo_             = data; hasCalo_             = true;}

      // get infos
      reco::TrackRef             trackRef          () const { return trackRef_;        }
      reco::MuonRef              muonRef           () const { return muonRef_;         }
      const reco::DeDxData&      dedxEstimator1    () const { return dedxEstim1_;      }
      const reco::DeDxData&      dedxEstimator2    () const { return dedxEstim2_;      }
      const reco::DeDxData&      dedxEstimator3    () const { return dedxEstim3_;      }
      const reco::DeDxData&      dedxEstimator4    () const { return dedxEstim4_;      }
      const reco::DeDxData&      dedxEstimator5    () const { return dedxEstim5_;      }
      const reco::DeDxData&      dedxEstimator6    () const { return dedxEstim6_;      }
      const reco::DeDxData&      dedxDiscriminator1() const { return dedxDiscrim1_;    }
      const reco::DeDxData&      dedxDiscriminator2() const { return dedxDiscrim2_;    }
      const reco::DeDxData&      dedxDiscriminator3() const { return dedxDiscrim3_;    }
      const reco::DeDxData&      dedxDiscriminator4() const { return dedxDiscrim4_;    }
      const reco::DeDxData&      dedxDiscriminator5() const { return dedxDiscrim5_;    }
      const reco::DeDxData&      dedxDiscriminator6() const { return dedxDiscrim6_;    }
      const reco::DeDxData&      dedx         (int i) const;
      const reco::MuonTimeExtra& muonTimeDt        () const { return muonTimeDt_;      }
      const reco::MuonTimeExtra& muonTimeCsc       () const { return muonTimeCsc_;     }
      const reco::MuonTimeExtra& muonTimeCombined  () const { return muonTimeCombined_;}
      const RPCBetaMeasurement&  rpc               () const { return rpc_;             }
      const CaloBetaMeasurement& calo              () const { return calo_;            }

      // shortcut of long function
      float p ()   const;
      float pt()   const;
      int   type() const;

   private:
      bool hasMuonRef_;
      bool hasTrackRef_;
      bool hasDedxEstim1_;
      bool hasDedxEstim2_;
      bool hasDedxEstim3_;
      bool hasDedxEstim4_;
      bool hasDedxEstim5_;
      bool hasDedxEstim6_;
      bool hasDedxDiscrim1_;
      bool hasDedxDiscrim2_;
      bool hasDedxDiscrim3_;
      bool hasDedxDiscrim4_;
      bool hasDedxDiscrim5_;
      bool hasDedxDiscrim6_;
      bool hasMuonTimeDt_;
      bool hasMuonTimeCsc_;
      bool hasMuonTimeCombined_;
      bool hasRpc_;
      bool hasCalo_;

      reco::TrackRef      trackRef_;
      reco::MuonRef       muonRef_;
      reco::DeDxData      dedxEstim1_;
      reco::DeDxData      dedxEstim2_;
      reco::DeDxData      dedxEstim3_;
      reco::DeDxData      dedxEstim4_;
      reco::DeDxData      dedxEstim5_;
      reco::DeDxData      dedxEstim6_;
      reco::DeDxData      dedxDiscrim1_;
      reco::DeDxData      dedxDiscrim2_;
      reco::DeDxData      dedxDiscrim3_;
      reco::DeDxData      dedxDiscrim4_;
      reco::DeDxData      dedxDiscrim5_;
      reco::DeDxData      dedxDiscrim6_;
      reco::MuonTimeExtra muonTimeDt_;
      reco::MuonTimeExtra muonTimeCsc_;
      reco::MuonTimeExtra muonTimeCombined_;
      RPCBetaMeasurement  rpc_;
      CaloBetaMeasurement calo_;
  };

  typedef  std::vector<HSCParticle> HSCParticleCollection;
  typedef  edm::Ref<HSCParticleCollection> HSCParticleRef;
  typedef  edm::RefProd<HSCParticleCollection> HSCParticleRefProd;
  typedef  edm::RefVector<HSCParticleCollection> HSCParticleRefVector;
  typedef  edm::AssociationMap<edm::OneToOne<reco::TrackCollection, EcalRecHitCollection> > TracksEcalRecHitsMap;
  
}

#endif
