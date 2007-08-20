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

  int numberOfLeptons() const;
  int numberOfBQuarks() const;
  
  const reco::Candidate* singleLepton() const;
  const reco::Candidate* singleNeutrino() const;
  const reco::Candidate* singleW() const;
  const reco::Candidate* singleTop() const;
  const reco::Candidate* decayB() const;
  const reco::Candidate* associatedB() const;
  //  const reco::Candidate* recoilQuark() const;
  std::vector<const reco::Candidate*> lightQuarks(bool plusB=false) const;
  
 private:
  
  bool isLepton(const reco::Candidate& part) const 
  {return (abs(part.pdgId())==11 || abs(part.pdgId())==13 || abs(part.pdgId())==15);}
  bool isNeutrino(const reco::Candidate& part) const 
  {return (abs(part.pdgId())==12 || abs(part.pdgId())==14 || abs(part.pdgId())==16);}
  double flavour(const reco::Candidate& part) const 
  {return (double)(part.pdgId() / abs(part.pdgId()) );}
};


#endif
