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

namespace susybsm {

 class CaloBetaMeasurement
  {
   public:
     float hcalenergy, ecalenergy, hoenergy;
     float ecal3by3dir, ecal5by5dir;
     float hcal3by3dir, hcal5by5dir;
     float trkisodr;
     float ecaltime, ecalbeta;
  };
 
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
 
 class TimeMeasurement
  {
   public:
     bool isLeft;
     bool isPhi;
     float posInLayer;
     float distIP;
     int station;
     DetId driftCell;
  };

 class DriftTubeTOF
  {
   public:
     float invBeta;
     float invBetaErr;
     float invBetaFree;
     float invBetaFreeErr;
     float vertexTime;
     float vertexTimeErr;
     int nStations;
     int nHits;
     std::vector<TimeMeasurement> timeMeasurements;
  };

 typedef  edm::AssociationVector<reco::MuonRefProd,std::vector<DriftTubeTOF> >  MuonTOFCollection;
 typedef  MuonTOFCollection::value_type MuonTOF;
 typedef  edm::Ref<MuonTOFCollection> MuonTOFRef;
 typedef  edm::RefProd<MuonTOFCollection> MuonTOFRefProd;
 typedef  edm::RefVector<MuonTOFCollection> MuonTOFRefVector;

 class DeDxBeta 
  {
   public:
      DeDxBeta() { nDeDxHits_ = 0; }
      DeDxBeta(const reco::TrackRef& tk, float dedx, float dedxe, float k, int hits):
        track_(tk),dedx_(dedx),dedxerr_(dedxe),k_(k),nDeDxHits_(hits) {}
      DeDxBeta(const reco::TrackRef& tk, const reco::DeDxData& data, float k):track_(tk),k_(k) {
        dedx_ = data.dEdx();
	dedxerr_ = data.dEdxError();
	nDeDxHits_ = data.numberOfMeasurements();
      }
      float dedx() const { return dedx_; }
      float dedxError() const { return dedxerr_; }
      float invBeta2() const { return k_*dedx(); }
      float invBeta2err() const { return k_*dedxError(); }
      float beta() const { return 1./sqrt(k_*dedx()); }
      float nDedxHits() const { return nDeDxHits_; }
      reco::TrackRef track() const { return track_; }

   private:
      reco::TrackRef track_;
      float dedx_;
      float dedxerr_;
      float k_;
      int nDeDxHits_;
  }; 

 typedef std::vector<DeDxBeta> DeDxBetaCollection;

 class HSCParticle 
  {
   public:
      // constructor
      HSCParticle():hasCalo(false),hasRpc(false),hasDt(false),hasTk(false),isMuon_(false), isTrack_(false) {}

      // check available infos
      bool  isMuon()      const { return isMuon_;    }
      bool  isTrack()     const { return isTrack_;   }


      // set infos
      void  setMuon  (reco::MuonRef   data)  { muon_  = data; isMuon_  = true; }
      void  setTrack (reco::TrackRef  data)  { track_ = data; isTrack_ = true; }

      // get infos
      reco::TrackRef getTrack()              { return track_; }
      reco::MuonRef  getMuon ()              { return muon_ ; }


      // check available infos
      bool  hasCaloInfo() const { return hasCalo; }
      bool  hasRpcInfo()  const { return hasRpc;  }
      bool  hasDtInfo()   const { return hasDt;   }
      bool  hasTkInfo()   const { return hasTk;   }
      bool  emptyDTInfo() const { return  ( ! hasDt ) || (dt.nHits ==0  && dt.nStations ==0) ;  }
      // set infos
      void  setCalo(const CaloBetaMeasurement& data) { calo = data; hasCalo = true; }
      void  setRpc(const RPCBetaMeasurement& data)   { rpc = data; hasRpc = true; }
      void  setDt(const DriftTubeTOF& data)          { dt = data; hasDt = true; }
      void  setTk(const DeDxBeta& data)              { tk = data; hasTk = true; }
      // get infos
      const CaloBetaMeasurement& Calo() const { return calo; }
      const RPCBetaMeasurement&  Rpc()  const { return rpc; }
      const DriftTubeTOF&        Dt()   const { return dt; }
      const DeDxBeta&            Tk()   const { return tk; }
      CaloBetaMeasurement&       Calo() { return calo; }
      RPCBetaMeasurement&        Rpc()  { return rpc; }
      DriftTubeTOF&              Dt()   { return dt; }
      DeDxBeta&                  Tk()   { return tk; }
      // physical quantities
      float p()  const;
      float pt() const;
      float massTk()         const { return hasTkInfo() ? track_->p()*sqrt(tk.invBeta2()-1) : 0.; }
      float massTkError()    const;
      float massDt()         const { return hasDtInfo() ? muon_->track()->p()*sqrt(dt.invBeta*dt.invBeta-1) : 0.;}
      float massDtError()    const;
      float massAvg()        const { return (massTk()+massDt())/2.;}
      float massAvgError()   const;
      float massDtComb()     const { return hasDtInfo() ? muon_->combinedMuon()->p()*sqrt(dt.invBeta*dt.invBeta-1) : 0.;}
      float massDtSta()      const { return hasDtInfo() ? muon_->standAloneMuon()->p()*sqrt(dt.invBeta*dt.invBeta-1) : 0.;}
      float massDtAssoTk()   const { return hasTkInfo()&&hasDtInfo() ? tk.track()->p()*sqrt(dt.invBeta*dt.invBeta-1) : 0.;}
      float massDtBest()     const { return hasDtInfo() ? p()*sqrt(dt.invBeta*dt.invBeta-1) : 0.;}
      float massTkAssoComb() const { return hasTkInfo()&&hasDtInfo() ? muon_->combinedMuon()->p()*sqrt(tk.invBeta2()-1) : 0.;}
      float betaTk()         const { return hasTkInfo() ? 1/sqrt(tk.invBeta2()) : 1.; }
      float betaDt()         const { return hasDtInfo() ? 1/dt.invBeta : 1.; }
      float betaAvg()        const { return 2./(sqrt(tk.invBeta2())+dt.invBeta); }
      float compatibility()  const { return (sqrt(tk.invBeta2())-dt.invBeta)/
                                            (sqrt(tk.invBeta2()/tk.nDedxHits())*0.1+dt.invBeta*0.1) ;}
      float betaEcal()       const { return hasCaloInfo() ? Calo().ecalbeta : 1.; }
      float betaRpc()        const { return hasRpcInfo() ? Rpc().beta : 1.; }
      // check if tracks are available
      bool  hasMuonTrack()         const { return hasDt && muon_->track().isNonnull(); }
      bool  hasMuonStaTrack()      const { return hasDt && muon_->standAloneMuon().isNonnull(); }
      bool  hasMuonCombinedTrack() const { return hasDt && muon_->combinedMuon().isNonnull(); }
      bool  hasTrackerTrack()      const { return hasTk; }
      // retreive the tracks (! no implicit protection. Call methods above first.)
      const reco::Track& muonTrack()     const { return *muon_->track(); } 
      const reco::Track& staTrack()      const { return *muon_->standAloneMuon(); } 
      const reco::Track& combinedTrack() const { return *muon_->combinedMuon(); } 
      const reco::Track& trackerTrack()  const { return *track_; } 
   private:
      bool hasCalo;
      bool hasRpc;
      bool hasDt;
      bool hasTk;
      bool isMuon_;
      bool isTrack_;
      reco::TrackRef track_;
      reco::MuonRef  muon_;
      DriftTubeTOF  dt;
      DeDxBeta tk;
      RPCBetaMeasurement rpc;
      CaloBetaMeasurement calo;
  };

  typedef  std::vector<HSCParticle> HSCParticleCollection;
  typedef  edm::Ref<HSCParticleCollection> HSCParticleRef;
  typedef  edm::RefProd<HSCParticleCollection> HSCParticleRefProd;
  typedef  edm::RefVector<HSCParticleCollection> HSCParticleRefVector;
  typedef  edm::AssociationMap<edm::OneToOne<reco::TrackCollection, EcalRecHitCollection> > TracksEcalRecHitsMap;
  
}

#endif
