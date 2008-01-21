#ifndef HSCParticle_H
#define HSCParticle_H
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
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

class DeDxBeta {
   public:
      reco::TrackRef track;
      float invBeta2;
      float invBeta2Fit; 
      int nDeDxHits;
}; 


 class HSCParticle
  {
   public:
      MuonTOF  dt;
      bool hasDt;
      DeDxBeta tk;
      bool hasTk;
      float massTk()const {return tk.track->p()*sqrt(tk.invBeta2-1);}
      float massDt()const {return dt.first->track()->p()*sqrt(dt.second.invBeta*dt.second.invBeta-1);}
      float massDtComb()const {return dt.first->combinedMuon()->p()*sqrt(dt.second.invBeta*dt.second.invBeta-1);}
      float massDtSta()const {return dt.first->standAloneMuon()->p()*sqrt(dt.second.invBeta*dt.second.invBeta-1);}
      float massDtAssoTk()const {return tk.track->p()*sqrt(dt.second.invBeta*dt.second.invBeta-1);}
      float massTkAssoComb() const {return dt.first->combinedMuon()->p()*sqrt(tk.invBeta2-1);}
      float compatibility() const {return (sqrt(tk.invBeta2)-dt.second.invBeta)/(sqrt(tk.invBeta2/tk.nDeDxHits)*0.1+dt.second.invBeta*0.1);}
      bool  hasMuonTrack() const {return dt.first->track().isNonnull(); }
      const reco::Track & muonTrack() const {return *dt.first->track(); } 
      const reco::Track & staTrack() const {return *dt.first->standAloneMuon(); } 
      const reco::Track & combinedTrack() const {return *dt.first->combinedMuon(); } 
      const reco::Track & trackerTrack() const {return *tk.track; } 
      bool  emptyDTInfo() const { return  ( ! hasDt ) || (dt.second.nHits ==0  && dt.second.nStations ==0) ;  }
 };

typedef  std::vector<HSCParticle> HSCParticleCollection;
typedef  edm::Ref<HSCParticleCollection> HSCParticleRef;
typedef  edm::RefProd<HSCParticleCollection> HSCParticleRefProd;
typedef  edm::RefVector<HSCParticleCollection> HSCParticleRefVector;
 
 
}

#endif
