#ifndef TopObjects_TopGenEvent_h
#define TopObjects_TopGenEvent_h
#include "DataFormats/Candidate/interface/Candidate.h"

class TopGenEvent {

 public:

  TopGenEvent(){};
  TopGenEvent(reco::CandidateRefProd&, std::vector<const reco::Candidate*>);
  virtual ~TopGenEvent(){};

  const reco::CandidateCollection& particles() const { return *parts_; }
  const std::vector<const reco::Candidate*> initialPartons() const { return initPartons_;}
  const reco::Candidate* candidate(int) const;
  
  //common getters
  const reco::Candidate* eMinus() const   { return candidate( 11 );}
  const reco::Candidate* ePlus() const    { return candidate(-11 );}
  const reco::Candidate* muMinus() const  { return candidate( 13 );}
  const reco::Candidate* muPlus() const   { return candidate(-13 );}
  const reco::Candidate* tauMinus() const { return candidate( 15 );}
  const reco::Candidate* tauPlus() const  { return candidate(-15 );}
  const reco::Candidate* wMinus() const   { return candidate( 24 );}
  const reco::Candidate* wPlus() const    { return candidate(-24 );}
  const reco::Candidate* top() const      { return candidate(  6 );}
  const reco::Candidate* topBar() const   { return candidate( -6 );}
  const reco::Candidate* b() const        { return candidate(  5 );}
  const reco::Candidate* bBar() const     { return candidate( -5 );}

 protected:
  
  bool isLepton(const reco::Candidate& part) const 
  {return (abs(part.pdgId())==11 || abs(part.pdgId())==13 || abs(part.pdgId())==15);}
  bool isNeutrino(const reco::Candidate& part) const 
  {return (abs(part.pdgId())==12 || abs(part.pdgId())==14 || abs(part.pdgId())==16);}
  double flavour(const reco::Candidate& part) const 
  {return (double)(part.pdgId() / abs(part.pdgId()) );}
  
 protected:

  reco::CandidateRefProd parts_;                    //top decay chain
  std::vector<const reco::Candidate*> initPartons_; //initial partons
};

#endif
