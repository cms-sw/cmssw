#ifndef L1TauAnalyzer_MCTauCand_h
#define L1TauAnalyzer_MCTauCand_h

#include "CLHEP/HepMC/GenParticle.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "HepMC/SimpleVector.h"

class MCTauCand : public HepMC::GenParticle {
 public:
  MCTauCand();
  MCTauCand(const HepMC::GenParticle & c);
  MCTauCand(const HepMC::GenParticle & c, int decayM, int nprong);
  MCTauCand(const HepMC::GenParticle & c, int decayM, int nprong,
      const CLHEP::HepLorentzVector vP4);
  ~MCTauCand();

  int getDecayMode() { return _decayMode; } // 1: elec, 2: muon, 3: had
  int getnProng() const { return _nProng; }
  CLHEP::HepLorentzVector getVisibleP4() const { return _visibleP4; }

  void setDecayMode(int d) { _decayMode = d; }
  void setnProng(int n) { _nProng = n; }
  void setVisibleP4(CLHEP::HepLorentzVector p4) { _visibleP4 = p4; }

  void calcVisibleP4(CLHEP::HepLorentzVector nuP4) {
    CLHEP::HepLorentzVector m = CLHEP::HepLorentzVector( momentum().x(), momentum().y(),
							 momentum().z(), momentum().t()); 
    _visibleP4 = m - nuP4;
  }

  std::vector<HepMC::GenParticle> &       getStableHadronicDaughters()       { return _stableHadronicDaughters; }
  const std::vector<HepMC::GenParticle> & getStableHadronicDaughters() const { return _stableHadronicDaughters; }

 private:
  int _decayMode;
  int _nProng;
  std::vector<HepMC::GenParticle>   _stableHadronicDaughters;

  CLHEP::HepLorentzVector _visibleP4;

};

#endif
