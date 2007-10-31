#ifndef TopObjects_TopGenEvent_h
#define TopObjects_TopGenEvent_h
#include "DataFormats/Candidate/interface/Candidate.h"

class TopGenEvent {

 public:

  TopGenEvent(){};
  TopGenEvent(reco::CandidateRefProd&, reco::CandidateRefProd&);
  virtual ~TopGenEvent(){};

  const reco::CandidateCollection& particles() const { return *parts_; }
  const reco::CandidateCollection& initialPartons() const { return *initPartons_;}
  std::vector<const reco::Candidate*> lightQuarks(bool plusB=false) const;
  const reco::Candidate* candidate(int) const;

  int numberOfLeptons() const;
  int numberOfBQuarks() const;

  const reco::Candidate* singleLepton() const;
  const reco::Candidate* singleNeutrino() const;

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

  reco::CandidateRefProd parts_;       //top decay chain
  reco::CandidateRefProd initPartons_; //initial partons
};

#endif
