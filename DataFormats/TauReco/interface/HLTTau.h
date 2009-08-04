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
      eta_ = 0.;
      phi_ = 0.;
      pt_ = -1.;
      emIsolation_ = -1000.;
      leadTrackPtL25_ = 0.;
      leadPionPtL25_ = 0.;
      ptL25_ = -1.;
      trackIsolationL3_ = -1;
      nTrackIsolationL3_ = -1;
      sumPtTracksL3_ = -1000.;     
      seedEcalHitEt_ = -1.;
      ecalClusterShape_.clear();
      nEcalHits_ = -1;
      hcalIsolEt_ =-1.;
      seedHcalHitEt_ =-1.;
      hcalClusterShape_.clear();
      nHcalHits_=-1;
    }

    HLTTau(float eta, float phi, float pt) {
      eta_ = eta;
      phi_ = phi;
      pt_ = pt;
    }

    virtual ~HLTTau() { }

    float getEta() const { return eta_; }
    float getPhi() const { return phi_; }
    float getPt() const { return pt_; }

    float getEMIsolationValue() const { return emIsolation_;}
    double getSeedEcalHitEt() const {return seedEcalHitEt_;} //Lead PF Cluster Et /or simple cluster/or crystal
    std::vector<double> getEcalClusterShape() const {return ecalClusterShape_;} //cluster shapes eta [0], Phi[0] DeltaR [1]
    int getNEcalHits() const {return nEcalHits_;} //N Ecal PF Clusters or simple clusters or crystals
    double getHcalIsolEt() const {return hcalIsolEt_;}
    double getSeedHcalHitEt() const {return seedHcalHitEt_;}
    std::vector<double> getHcalClusterShape() const {return hcalClusterShape_;}
    int getNHcalHits() const {return nHcalHits_;}
    
    float getL25TauPt() const {return ptL25_;}
    float getL25LeadTrackPtValue() const { return leadTrackPtL25_; }
    float getL25LeadPionPtValue() const { return leadPionPtL25_; }
    int   getL3TrackIsolationResponse()const { return trackIsolationL3_; }
    int   getNL3TrackIsolation() const { return nTrackIsolationL3_; }
    float getSumPtTracksL3() const {return sumPtTracksL3_;}


    //L2 quantities
    void setEMIsolationValue(double emiso) { emIsolation_ = emiso;}
    void setSeedEcalHitEt(double  seed)   {seedEcalHitEt_ = seed;} 
    void setEcalClusterShape(std::vector<double> clusters)  {ecalClusterShape_ = clusters;} 
    void setNEcalHits(int nhits)  { nEcalHits_ = nhits;}     
    void setHcalIsolEt(double hcalIso)  { hcalIsolEt_ = hcalIso;}
    void setSeedHcalHitEt(double seed)  { seedHcalHitEt_ = seed;}
    void setHcalClusterShape(std::vector<double> clusters)  { hcalClusterShape_ = clusters;}
    void setNHcalHits(int nhits)  { nHcalHits_ = nhits;}
    
    //L2.5 quantities
    void setL25TauPt(double pt)  { ptL25_ = pt;}
    void setL25LeadTrackPtValue(double pt)  {  leadTrackPtL25_ = pt; }
    void setL25LeadPionPtValue(double pt)  {  leadPionPtL25_ = pt; }

    //L3 quantities
    void setL3TrackIsolationResponse(int response) {  trackIsolationL3_ = response; }
    void setNL3TrackIsolation(int nTracks)  { nTrackIsolationL3_ = nTracks; }  
    void setSumPtTracksL3(double sumPt) {sumPtTracksL3_ = sumPt;}    
   






  private:
    float eta_, phi_, pt_;
    float emIsolation_;
    float ptL25_;
    float leadTrackPtL25_;
    float leadPionPtL25_;
    int   trackIsolationL3_;
    int nTrackIsolationL3_;
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
