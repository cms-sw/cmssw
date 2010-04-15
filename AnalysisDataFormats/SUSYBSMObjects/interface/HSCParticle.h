#ifndef HSCParticle_H
#define HSCParticle_H
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include <vector>

#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"

namespace susybsm {
 
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
     float hcalenergy, ecalenergy, hoenergy;
     float ecal3by3dir, ecal5by5dir;
     float hcal3by3dir, hcal5by5dir;
     float trkisodr;
     float ecaltime, ecalbeta;
  };

 class HSCParticle 
  {
   public:
      // constructor
      HSCParticle(){
         hasMuonRef_          = false;
         hasTrackRef_         = false;
         hasDedxEstim_        = false;
         hasDedxDiscrim_      = false;
         hasMuonTimeDt_       = false;
         hasMuonTimeCsc_      = false;
         hasMuonTimeCombined_ = false;
         hasRpc_              = false;
         hasCalo_             = false;
      }

      // check available infos
      bool  hasMuonRef()          const { return hasMuonRef_;          }
      bool  hasTrackRef()         const { return hasTrackRef_;         }
      bool  hasDedxEstim()        const { return hasDedxEstim_;        }
      bool  hasDedxDiscrim()      const { return hasDedxDiscrim_;      }
      bool  hasMuonTimeDt()       const { return hasMuonTimeDt_;       }
      bool  hasMuonTimeCsc()      const { return hasMuonTimeCsc_;      }
      bool  hasMuonTimeCombined() const { return hasMuonTimeCombined_; }
      bool  hasRpcInfo()          const { return hasRpc_;              }
      bool  hasCaloInfo()         const { return hasCalo_;             }

      // set infos
      void setMuon             (const reco::MuonRef&       data) {muonRef_          = data; hasMuonRef_          = true;}
      void setTrack            (const reco::TrackRef&      data) {trackRef_         = data; hasTrackRef_         = true;}
      void setDedxEstimator    (const reco::DeDxData&      data) {dedxEstim_        = data; hasDedxEstim_        = true;}
      void setDedxDiscriminator(const reco::DeDxData&      data) {dedxDiscrim_      = data; hasDedxDiscrim_      = true;}
      void setMuonTimeDt       (const reco::MuonTimeExtra& data) {muonTimeDt_       = data; hasMuonTimeDt_       = true;}
      void setMuonTimeCsc      (const reco::MuonTimeExtra& data) {muonTimeCsc_      = data; hasMuonTimeCsc_      = true;}
      void setMuonTimeCombined (const reco::MuonTimeExtra& data) {muonTimeCombined_ = data; hasMuonTimeCombined_ = true;}
      void setRpc              (const RPCBetaMeasurement&  data) {rpc_              = data; hasRpc_              = true;}
      void setCalo             (const CaloBetaMeasurement& data) {calo_             = data; hasCalo_             = true;}

      // get infos
      reco::TrackRef             trackRef         () const { return trackRef_;        }
      reco::MuonRef              muonRef          () const { return muonRef_;         }
      const reco::DeDxData&      dedxEstimator    () const { return dedxEstim_;       }
      const reco::DeDxData&      dedxDiscriminator() const { return dedxDiscrim_;     }
      const reco::MuonTimeExtra& muonTimeDt       () const { return muonTimeDt_;      }
      const reco::MuonTimeExtra& muonTimeCsc      () const { return muonTimeCsc_;     }
      const reco::MuonTimeExtra& muonTimeCombined () const { return muonTimeCombined_;}
      const RPCBetaMeasurement&  rpc              () const { return rpc_;             }
      const CaloBetaMeasurement& calo             () const { return calo_;            }

      // shortcut of long function
      float p ()  const;
      float pt()  const;

   private:
      bool hasMuonRef_;
      bool hasTrackRef_;
      bool hasDedxEstim_;
      bool hasDedxDiscrim_;
      bool hasMuonTimeDt_;
      bool hasMuonTimeCsc_;
      bool hasMuonTimeCombined_;
      bool hasRpc_;
      bool hasCalo_;

      reco::TrackRef      trackRef_;
      reco::MuonRef       muonRef_;
      reco::DeDxData      dedxEstim_;
      reco::DeDxData      dedxDiscrim_;
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
