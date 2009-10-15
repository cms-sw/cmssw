#ifndef __BaseHiGenEvtSelector_h_
#define __BaseHiGenEvtSelector_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

class BaseHiGenEvtSelector {
 public:
   BaseHiGenEvtSelector(const edm::ParameterSet&){;}
   virtual ~BaseHiGenEvtSelector(){;}
   virtual bool filter(HepMC::GenEvent *){return true;}
   bool selectParticle(HepMC::GenParticle* par, int status, int pdg /*Absolute*/, double ptMin, double etaMax){
      return (par->status() == status && abs(par->pdg_id()) == pdg && par->momentum().perp() > ptMin && fabs(par->momentum().eta()) < etaMax);
   }
};

#endif

