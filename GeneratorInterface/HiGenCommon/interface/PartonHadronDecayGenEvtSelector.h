#ifndef _HI_PartonHadronDecayGenEvtSelector_h__
#define _HI_PartonHadronDecayGenEvtSelector_h__

#include <vector>
#include "GeneratorInterface/HiGenCommon/interface/BaseHiGenEvtSelector.h"

class PartonHadronDecayGenEvtSelector : public BaseHiGenEvtSelector 
{
 public:
  PartonHadronDecayGenEvtSelector(const edm::ParameterSet& pset);
  virtual ~PartonHadronDecayGenEvtSelector(){;}
  
  bool filter(HepMC::GenEvent *);
  bool selectParticle(HepMC::GenParticle* par, int status, int pdg /*Absolute*/, double ptMin, double etaMax){
    return (par->status() == status && abs(par->pdg_id()) == pdg && par->momentum().perp() > ptMin && fabs(par->momentum().eta()) < etaMax);
  }
  bool selectParticle(HepMC::GenParticle* par, int status, int pdg /*Absolute*/, double etaMax, double etaMin, double pMin, double ptMax, double ptMin){
    return (par->status() == status &&
	    abs(par->pdg_id()) == pdg &&
	    par->momentum().eta() < etaMax && par->momentum().eta() > etaMin &&
	    par->momentum().rho() > pMin &&
	    par->momentum().perp() < ptMax && par->momentum().perp() > ptMin);
  }

 private:

  std::vector<int>    partonId_;
  std::vector<int>    partonStatus_;
  std::vector<double> partonEtaMax_;
  std::vector<double> partonPtMin_;

  std::vector<int>    hadronId_;
  std::vector<int>    hadronStatus_;
  std::vector<double> hadronEtaMax_;
  std::vector<double> hadronEtaMin_;
  std::vector<double> hadronPMin_;
  std::vector<double> hadronPtMax_;
  std::vector<double> hadronPtMin_;

  int    decayId_;
  int    decayStatus_;
  double decayEtaMax_;
  double decayEtaMin_;
  double decayPMin_;
  double decayPtMax_;
  double decayPtMin_;
  int    decayNtrig_;

};

#endif
