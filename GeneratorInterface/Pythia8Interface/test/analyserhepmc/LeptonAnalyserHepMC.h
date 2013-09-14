#include "HepMC/GenEvent.h"
#include <fastjet/PseudoJet.hh>


#include <vector>

class LeptonAnalyserHepMC {

public:
  LeptonAnalyserHepMC (double aMaxEta = 2.4, double aThresholdEt = 20.);

  std::vector<HepMC::GenParticle> isolatedLeptons(const HepMC::GenEvent* pEv);
  int nIsolatedLeptons(const HepMC::GenEvent* pEv);
  double MinMass(const HepMC::GenEvent* pEv);
  std::vector <fastjet::PseudoJet>
    removeLeptonsFromJets(std::vector<fastjet::PseudoJet>& jets,
                          const HepMC::GenEvent* pEv);

private:
  double MaxEta;
  double ThresholdEt;
  double RConeIsol;
  double MaxPtIsol;
  double RIdJet;
  double EpsIdJet;
};
