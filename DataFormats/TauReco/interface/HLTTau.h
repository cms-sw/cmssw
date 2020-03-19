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
      ecalClusterShape_.clear();
      hcalClusterShape_.clear();
    }

    HLTTau(float eta,
           float phi,
           float pt,
           float emIsolation,
           int trackIsolationL25,
           float leadTrackPtL25,
           int trackIsolationL3,
           float leadTrackPtL3) {
      eta_ = eta;
      phi_ = phi;
      pt_ = pt;
      emIsolation_ = emIsolation;
      trackIsolationL25_ = trackIsolationL25;
      leadTrackPtL25_ = leadTrackPtL25;
      trackIsolationL3_ = trackIsolationL3;
      leadTrackPtL3_ = leadTrackPtL3;
    }

    virtual ~HLTTau() {}

    float getEta() const { return eta_; }
    float getPhi() const { return phi_; }
    float getPt() const { return pt_; }

    float getEMIsolationValue() const { return emIsolation_; }
    int getL25TrackIsolationResponse() const { return trackIsolationL25_; }
    int getNL25TrackIsolation() const { return nTrackIsolationL25_; }
    float getL25LeadTrackPtValue() const { return leadTrackPtL25_; }
    int getL3TrackIsolationResponse() const { return trackIsolationL3_; }
    int getNL3TrackIsolation() const { return nTrackIsolationL3_; }
    float getL3LeadTrackPtValue() const { return leadTrackPtL3_; }
    float getSumPtTracksL25() const { return sumPtTracksL25_; }
    float getSumPtTracksL3() const { return sumPtTracksL3_; }

    double getSeedEcalHitEt() const { return seedEcalHitEt_; }  //Lead PF Cluster Et /or simple cluster/or crystal
    std::vector<double> getEcalClusterShape() const {
      return ecalClusterShape_;
    }                                                //cluster shapes eta [0], Phi[0] DeltaR [1]
    int getNEcalHits() const { return nEcalHits_; }  //N Ecal PF Clusters or simple clusters or crystals

    double getHcalIsolEt() const { return hcalIsolEt_; }
    double getSeedHcalHitEt() const { return seedHcalHitEt_; }
    std::vector<double> getHcalClusterShape() const { return hcalClusterShape_; }
    int getNHcalHits() const { return nHcalHits_; }

    void setNL25TrackIsolation(int nTracks) { nTrackIsolationL25_ = nTracks; }
    void setNL3TrackIsolation(int nTracks) { nTrackIsolationL3_ = nTracks; }
    void setSumPtTracksL25(double sumPt) { sumPtTracksL25_ = sumPt; }
    void setSumPtTracksL3(double sumPt) { sumPtTracksL3_ = sumPt; }
    void setSeedEcalHitEt(double seed) { seedEcalHitEt_ = seed; }
    void setEcalClusterShape(const std::vector<double>& clusters) { ecalClusterShape_ = clusters; }
    void setNEcalHits(int nhits) { nEcalHits_ = nhits; }

    void setHcalIsolEt(double hcalIso) { hcalIsolEt_ = hcalIso; }
    void setSeedHcalHitEt(double seed) { seedHcalHitEt_ = seed; }
    void setHcalClusterShape(const std::vector<double>& clusters) { hcalClusterShape_ = clusters; }
    void setNHcalHits(int nhits) { nHcalHits_ = nhits; }

  private:
    float eta_ = 0., phi_ = 0., pt_ = -1.;
    float emIsolation_ = -1000.;
    int trackIsolationL25_ = -1;
    float leadTrackPtL25_ = 0.;
    int nTrackIsolationL25_ = -1;
    int trackIsolationL3_ = -1;
    int nTrackIsolationL3_ = -1;
    float leadTrackPtL3_ = 0.;
    double seedEcalHitEt_ = -1;
    std::vector<double> ecalClusterShape_;
    int nEcalHits_ = -1;
    double hcalIsolEt_ = -1;
    double seedHcalHitEt_ = -1;
    std::vector<double> hcalClusterShape_;
    int nHcalHits_ = -1;
    double sumPtTracksL25_ = -1000.;
    double sumPtTracksL3_ = -1000.;
  };

}  // namespace reco
#endif
