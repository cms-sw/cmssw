#ifndef __BaseHiGenSkimmer_h_
#define __BaseHiGenSkimmer_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

class BaseHiGenSkimmer {
 public:
   BaseHiGenSkimmer(const edm::ParameterSet&){;}
   virtual ~BaseHiGenSkimmer(){;}
   virtual bool filter(HepMC::GenEvent *){return true;}
   bool selectParticle(HepMC::GenParticle* par, int status, int pdg /*Absolute*/, double ptMin, double etaMax){
      return (par->status() == status && abs(par->pdg_id()) == pdg && par->momentum().perp() > ptMin && fabs(par->momentum().eta()) < etaMax);
   }
};

#endif

