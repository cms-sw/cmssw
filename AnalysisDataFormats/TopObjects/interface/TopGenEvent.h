#ifndef TopObjects_TopGenEvent_h
#define TopObjects_TopGenEvent_h
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

namespace TopDecayID{
  static const int stable = 2;
  static const int unfrag = 3;
  static const int tID    = 6;
  static const int bID    = 5;
  static const int glueID = 21;
  static const int photID = 22;
  static const int ZID    = 23;
  static const int WID    = 24;
  static const int tauID  = 15;
}

// Now you can access to each generated particle for differents status
// status 2 -> after all radiations (top & other particles' radiations) : stable (came be compare to previous versions of TopDecaySubset)
// status 3 -> before radiation : unfrag (intermediate state - directly coming from genParticles collection)
// status 4 -> resum the quark status 2 with its own radiations (taking properly into account top radiation) (New Version)
// To change the default status (which is 4) you just have to do setDefaultStatus(new_status)

class TopGenEvent {

 public:

  TopGenEvent(){};
  TopGenEvent(reco::GenParticleRefProd&, reco::GenParticleRefProd&, int status=4);
  virtual ~TopGenEvent(){};

  void setDefaultStatus(int status=4){ defaultStatus_=status;};
  void dumpEventContent() const;
  const reco::GenParticleCollection& particles() const { return *parts_; }
  const reco::GenParticleCollection& initialPartons() const { return *initPartons_;}
  std::vector<const reco::GenParticle*> lightQuarks(bool plusB=false) const;
  const reco::GenParticle* candidate(int pdgId) const;

  int numberOfLeptons() const;
  int numberOfLeptonsFromW() const;
  int numberOfBQuarks() const;
  int numberOfBQuarksFromTop() const;
  int numberOfISR() const {return ((int) ISR().size());};
  int numberOfTopsRadiation() const {return ((int) (topRadiation().size()+topBarRadiation().size()));};

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
  std::vector<const reco::GenParticle*> topRadiation() const;
  std::vector<const reco::GenParticle*> topBarRadiation() const;

  std::vector<const reco::GenParticle*> ISR() const;
 protected:

  int defaultStatus_; // default status for the reco::GenParticle* return (expect leptons where status == 3)
  reco::GenParticleRefProd parts_;       //top decay chain
  reco::GenParticleRefProd initPartons_; //initial partons
};

#endif
