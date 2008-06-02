// (Andrea G., 2007/07/27)
// This class deals with the case of t-channel single top. Different members should be written for s-channel and Wt.
// Only the leptonic decay is considered here.

#ifndef TopObjects_StGenEvent_h
#define TopObjects_StGenEvent_h

#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"

class StGenEvent: public TopGenEvent {

 public:
  StGenEvent();
  StGenEvent(reco::GenParticleRefProd&, reco::GenParticleRefProd&);
  virtual ~StGenEvent();

  const reco::GenParticle* singleW() const;
  const reco::GenParticle* singleTop() const;
  const reco::GenParticle* decayB() const;
  const reco::GenParticle* associatedB() const;
  //const reco::Candidate* recoilQuark() const;
  
 private:
  
};

#endif
