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
    HLTTau(){
      metCut_ = 0.;
      emIsolation_ = 0.;
      trackIsolationL25_ = 0;
      leadTrackPtL25_ = 0.;
      trackIsolationL3_ =0;
      leadTrackPtL3_ = 0.;
    }
    HLTTau(float metCut, float emIsolation, int trackIsolationL25, float leadTrackPtL25, int trackIsolationL3, float leadTrackPtL3) {
      metCut_ = metCut;
      emIsolation_ = emIsolation;
      trackIsolationL25_ = trackIsolationL25;
      leadTrackPtL25_ = leadTrackPtL25;
      trackIsolationL3_ =trackIsolationL3 ;
      leadTrackPtL3_ = leadTrackPtL3;
    }
    virtual ~HLTTau(){}

    float getMETValue()const { return metCut_;}
    float getEMIsolationValue() const { return emIsolation_;}
    int getL25TrackIsolationResponse()const { return trackIsolationL25_;}    
    float getL25LeadTrackPtValue() const { return leadTrackPtL25_;}
    int getL3TrackIsolationResponse()const { return trackIsolationL3_;} 
    float getL3LeadTrackPtValue() const { return leadTrackPtL3_;}



  private:
float    metCut_;
float emIsolation_;
int trackIsolationL25_;
float    leadTrackPtL25_;
int    trackIsolationL3_;
float    leadTrackPtL3_;

  };
}
#endif
