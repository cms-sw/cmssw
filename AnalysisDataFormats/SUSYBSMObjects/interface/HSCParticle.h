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
      HSCParticle():hasDt(false),hasTk(false) {}
      bool  hasDtInfo() const {return hasDt; }
      bool  hasTkInfo() const {return hasTk; }
      void  setDt(const MuonTOF& data)  { dt = data; hasDt = true; }
      void  setTk(const DeDxBeta& data) { tk = data; hasTk = true; }
      const MuonTOF& Dt() const { return dt; }
      const DeDxBeta& Tk() const { return tk; }
      MuonTOF& Dt() { return dt; }
      DeDxBeta& Tk() { return tk; }
      float p() const;
      float pt() const;
      float massTk()const {return tk.track()->p()*sqrt(tk.invBeta2()-1);}
      float massDt()const {return dt.first->track()->p()*sqrt(dt.second.invBeta*dt.second.invBeta-1);}
      float massDtComb()const {return dt.first->combinedMuon()->p()*sqrt(dt.second.invBeta*dt.second.invBeta-1);}
      float massDtSta()const {return dt.first->standAloneMuon()->p()*sqrt(dt.second.invBeta*dt.second.invBeta-1);}
      float massDtAssoTk()const {return tk.track()->p()*sqrt(dt.second.invBeta*dt.second.invBeta-1);}
      float massDtBest() const { return p()*sqrt(dt.second.invBeta*dt.second.invBeta-1);}
      float massTkAssoComb() const {return dt.first->combinedMuon()->p()*sqrt(tk.invBeta2()-1);}
      float compatibility() const {return (sqrt(tk.invBeta2())-dt.second.invBeta)/(sqrt(tk.invBeta2()/tk.nDedxHits())*0.1+dt.second.invBeta*0.1);}
      bool  hasMuonTrack() const {return dt.first->track().isNonnull(); }
      bool  hasMuonStaTrack() const {return dt.first->standAloneMuon().isNonnull(); }
      bool  hasMuonCombinedTrack() const {return dt.first->combinedMuon().isNonnull(); }
      bool  hasTrackerTrack() const {return hasTk; }
      const reco::Track& muonTrack() const {return *dt.first->track(); } 
      const reco::Track& staTrack() const {return *dt.first->standAloneMuon(); } 
      const reco::Track& combinedTrack() const {return *dt.first->combinedMuon(); } 
      const reco::Track& trackerTrack() const {return *tk.track(); } 
      bool  emptyDTInfo() const { return  ( ! hasDt ) || (dt.second.nHits ==0  && dt.second.nStations ==0) ;  }
   private:
      bool hasDt;
      bool hasTk;
      MuonTOF  dt;
      DeDxBeta tk;
  };

  typedef  std::vector<HSCParticle> HSCParticleCollection;
  typedef  edm::Ref<HSCParticleCollection> HSCParticleRef;
  typedef  edm::RefProd<HSCParticleCollection> HSCParticleRefProd;
  typedef  edm::RefVector<HSCParticleCollection> HSCParticleRefVector;
  typedef  edm::AssociationMap<edm::OneToOne<reco::TrackCollection, EcalRecHitCollection> > TracksEcalRecHitsMap;
  
}

#endif
