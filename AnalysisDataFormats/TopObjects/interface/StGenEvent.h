// (Andrea G., 2007/07/27)
// This class deals with the case of t-channel single top. Different members should be written for s-channel and Wt.
// Only the leptonic decay is considered here.

#ifndef TopObjects_StGenEvent_h
#define TopObjects_StGenEvent_h
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"

class StGenEvent: public TopGenEvent {

 public:
  
  StGenEvent();
  StGenEvent(reco::CandidateRefProd&, std::vector<const reco::Candidate*>);
  virtual ~StGenEvent();

  const reco::Candidate* singleW() const;
  const reco::Candidate* singleTop() const;
  const reco::Candidate* decayB() const;
  const reco::Candidate* associatedB() const;
  //  const reco::Candidate* recoilQuark() const;
  
 private:
  
};


#endif
