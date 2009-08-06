#ifndef TopObjects_TopGenEvent_h
#define TopObjects_TopGenEvent_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"


namespace TopDecayID{
  /// identification of top decays; used for following
  /// the decay chain in TopDecaySubset
  static const int stable = 2;
  static const int unfrag = 3;
  static const int tID    = 6;
  static const int bID    = 5;
  static const int glueID = 21;
  static const int photID = 22;
  static const int ZID    = 23;
  static const int WID    = 24;
  static const int elecID = 11;
  static const int muonID = 13;
  static const int tauID  = 15;
}

namespace WDecay{
  /// classification of leptons in the decay channel 
  /// of the W boson used in several places throughout 
  /// the package
  enum LeptonType {kNone, kElec, kMuon, kTau};
}

/**
   \class   TopGenEvent TopGenEvent.h "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"

   \brief   Base class to hold information for reduced top generator information

   The structure holds reference information to the generator particles 
   of the decay chains for each top quark and of the initial partons. It 
   provides access and administration.
*/

class TopGenEvent {

 public:

  /// empty constructor
  TopGenEvent(){};
  /// default constructor
  TopGenEvent(reco::GenParticleRefProd&, reco::GenParticleRefProd&);
  /// default destructor
  virtual ~TopGenEvent(){};

  /// return particles of decay chain
  const reco::GenParticleCollection& particles() const { return *parts_; }
  /// return particles of initial partons
  const reco::GenParticleCollection& initialPartons() const { return *initPartons_;}
  /// return radiated gluons from particle with pdgId
  std::vector<const reco::GenParticle*> radiatedGluons(int pdgId) const;
  /// return all light quarks or all quarks including b's 
  std::vector<const reco::GenParticle*> lightQuarks(bool plusB=false) const;
  /// return number of leptons in the decay chain
  int numberOfLeptons(bool fromWBoson=true) const;
  /// return number of leptons in the decay chain
  int numberOfLeptons(WDecay::LeptonType type, bool fromWBoson=true) const;
  /// return number of b quarks in the decay chain
  int numberOfBQuarks(bool fromTopQuark=true) const;
  /// return number of top anti-top sisters
  std::vector<const reco::GenParticle*> topSisters() const;

  /// get candidate with given pdg id if available; 0 else 
  const reco::GenParticle* candidate(int id) const;
  /// return electron if available; 0 else
  const reco::GenParticle* eMinus() const   { return candidate( TopDecayID::elecID );}
  /// return positron if available; 0 else
  const reco::GenParticle* ePlus() const    { return candidate(-TopDecayID::elecID );}
  /// return muon if available; 0 else
  const reco::GenParticle* muMinus() const  { return candidate( TopDecayID::muonID );}
  /// return anti-muon if available; 0 else
  const reco::GenParticle* muPlus() const   { return candidate(-TopDecayID::muonID );}
  /// return tau if available; 0 else
  const reco::GenParticle* tauMinus() const { return candidate( TopDecayID::tauID  );}
  /// return anti-tau if available; 0 else
  const reco::GenParticle* tauPlus() const  { return candidate(-TopDecayID::tauID  );}
  /// return W minus if available; 0 else
  const reco::GenParticle* wMinus() const   { return candidate( TopDecayID::WID    );}
  /// return W plus if available; 0 else
  const reco::GenParticle* wPlus() const    { return candidate(-TopDecayID::WID    );}
  /// return top if available; 0 else
  const reco::GenParticle* top() const      { return candidate( TopDecayID::tID    );}
  /// return anti-top if available; 0 else
  const reco::GenParticle* topBar() const   { return candidate(-TopDecayID::tID    );}
  /// return b quark if available; 0 else
  const reco::GenParticle* b() const        { return candidate( TopDecayID::bID    );}
  /// return anti-b quark if available; 0 else
  const reco::GenParticle* bBar() const     { return candidate(-TopDecayID::bID    );}

  /// print content of the top decay chain as formated 
  /// LogInfo to the MessageLogger output for debugging
  void print() const;


 protected:

  /// reference to the top decay chain (has to be kept in the event!)
  reco::GenParticleRefProd parts_;       
  /// reference to the list of initial partons (has to be kept in the event!)
  reco::GenParticleRefProd initPartons_; 
};

#endif
