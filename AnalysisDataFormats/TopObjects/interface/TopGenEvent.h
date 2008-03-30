#ifndef TopObjects_TopGenEvent_h
#define TopObjects_TopGenEvent_h
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class TopGenEvent {

 public:

  TopGenEvent(){};
  TopGenEvent(reco::GenParticleRefProd&, reco::GenParticleRefProd&);
  virtual ~TopGenEvent(){};

  void dumpEventContent() const;
  const reco::GenParticleCollection& particles() const { return *parts_; }
  const reco::GenParticleCollection& initialPartons() const { return *initPartons_;}
  std::vector<const reco::GenParticle*> lightQuarks(bool plusB=false) const;
  const reco::GenParticle* candidate(int) const;

  int numberOfLeptons() const;
  int numberOfBQuarks() const;

  const reco::GenParticle* singleLepton() const;
  const reco::GenParticle* singleNeutrino() const;

  const reco::GenParticle* eMinus() const   { return candidate( 11 );}
  const reco::GenParticle* ePlus() const    { return candidate(-11 );}
  const reco::GenParticle* muMinus() const  { return candidate( 13 );}
  const reco::GenParticle* muPlus() const   { return candidate(-13 );}
  const reco::GenParticle* tauMinus() const { return candidate( 15 );}
  const reco::GenParticle* tauPlus() const  { return candidate(-15 );}
  const reco::GenParticle* wMinus() const   { return candidate( 24 );}
  const reco::GenParticle* wPlus() const    { return candidate(-24 );}
  const reco::GenParticle* top() const      { return candidate(  6 );}
  const reco::GenParticle* topBar() const   { return candidate( -6 );}
  const reco::GenParticle* b() const        { return candidate(  5 );}
  const reco::GenParticle* bBar() const     { return candidate( -5 );}

 protected:

  reco::GenParticleRefProd parts_;       //top decay chain
  reco::GenParticleRefProd initPartons_; //initial partons
};

#endif
