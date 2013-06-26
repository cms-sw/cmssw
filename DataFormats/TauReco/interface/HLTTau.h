#ifndef DataFormats_TauReco_HLTTau_h
#define DataFormats_TauReco_HLTTau_h

/* class HLTTau
 * authors: Simone Gennai (simone.gennai@cern.ch)
 * created: Oct 25 2007,
 */

//Very simple class to be used in HLT-Open environment for rate computation.


#include "DataFormats/TauReco/interface/HLTTauFwd.h"

namespace reco {

  class HLTTau {
  public:
    HLTTau() {
      emIsolation_ = -1000.;
      sumPtTracksL25_ = -1000.;
      sumPtTracksL3_ = -1000.;
      trackIsolationL25_ = -1;
      nTrackIsolationL25_ = -1;
      leadTrackPtL25_ = 0.;
      trackIsolationL3_ = -1;
      nTrackIsolationL3_ = -1;
      leadTrackPtL3_ = 0.;
      eta_ = 0.;
      phi_ = 0.;
      pt_ = -1.;
      seedEcalHitEt_ = -1.;
      ecalClusterShape_.clear();
      nEcalHits_ = -1;
      hcalIsolEt_ =-1.;
      seedHcalHitEt_ =-1.;
      hcalClusterShape_.clear();
      nHcalHits_=-1;
    }

    HLTTau(float eta, float phi, float pt, float emIsolation, int trackIsolationL25, float leadTrackPtL25, int trackIsolationL3, float leadTrackPtL3) {
      eta_ = eta;
      phi_ = phi;
      pt_ = pt;
      emIsolation_ = emIsolation;
      trackIsolationL25_ = trackIsolationL25;
      leadTrackPtL25_ = leadTrackPtL25;
      trackIsolationL3_ = trackIsolationL3 ;
      leadTrackPtL3_ = leadTrackPtL3;
    }

    virtual ~HLTTau() { }

    float getEta() const { return eta_; }
    float getPhi() const { return phi_; }
    float getPt() const { return pt_; }

    float getEMIsolationValue() const { return emIsolation_;}
    int   getL25TrackIsolationResponse() const { return trackIsolationL25_; }
    int   getNL25TrackIsolation() const { return nTrackIsolationL25_; }
    float getL25LeadTrackPtValue() const { return leadTrackPtL25_; }
    int   getL3TrackIsolationResponse()const { return trackIsolationL3_; }
    int   getNL3TrackIsolation() const { return nTrackIsolationL3_; }
    float getL3LeadTrackPtValue() const { return leadTrackPtL3_; }
    float getSumPtTracksL25() const {return sumPtTracksL25_;}
    float getSumPtTracksL3() const {return sumPtTracksL3_;}

    double getSeedEcalHitEt() const {return seedEcalHitEt_;} //Lead PF Cluster Et /or simple cluster/or crystal
    std::vector<double> getEcalClusterShape() const {return ecalClusterShape_;} //cluster shapes eta [0], Phi[0] DeltaR [1]
    int getNEcalHits() const {return nEcalHits_;} //N Ecal PF Clusters or simple clusters or crystals
    
    double getHcalIsolEt() const {return hcalIsolEt_;}
    double getSeedHcalHitEt() const {return seedHcalHitEt_;}
    std::vector<double> getHcalClusterShape() const {return hcalClusterShape_;}
    int getNHcalHits() const {return nHcalHits_;}


    void  setNL25TrackIsolation(int nTracks)  { nTrackIsolationL25_ = nTracks; }
    void  setNL3TrackIsolation(int nTracks)  { nTrackIsolationL3_ = nTracks; }  
    void setSumPtTracksL25(double sumPt) {sumPtTracksL25_ = sumPt;}
    void setSumPtTracksL3(double sumPt) {sumPtTracksL3_ = sumPt;}    
    void setSeedEcalHitEt(double  seed)   {seedEcalHitEt_ = seed;} 
    void setEcalClusterShape(const std::vector<double>& clusters)  {ecalClusterShape_ = clusters;} 
    void setNEcalHits(int nhits)  { nEcalHits_ = nhits;} 
    
    void setHcalIsolEt(double hcalIso)  { hcalIsolEt_ = hcalIso;}
    void setSeedHcalHitEt(double seed)  { seedHcalHitEt_ = seed;}
    void setHcalClusterShape(const std::vector<double>& clusters)  { hcalClusterShape_ = clusters;}
    void setNHcalHits(int nhits)  { nHcalHits_ = nhits;}


  private:
    float eta_, phi_, pt_;
    float emIsolation_;
    int   trackIsolationL25_;
    float leadTrackPtL25_;
    int nTrackIsolationL25_;
    int   trackIsolationL3_;
    int nTrackIsolationL3_;
    float leadTrackPtL3_;
    double seedEcalHitEt_;
    std::vector<double> ecalClusterShape_;
    int nEcalHits_;
    double hcalIsolEt_;
    double seedHcalHitEt_;
    std::vector<double> hcalClusterShape_;
    int nHcalHits_;
    double sumPtTracksL25_;
    double sumPtTracksL3_;
  };
  
}
#endif
