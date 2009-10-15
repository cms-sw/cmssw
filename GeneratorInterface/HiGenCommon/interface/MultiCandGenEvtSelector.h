#ifndef _HI_DiMuonSkimmer_h__
#define _HI_DiMuonSkimmer_h__

#include "GeneratorInterface/HiGenCommon/interface/BaseHiGenEvtSelector.h"

class MultiCandGenEvtSelector : public BaseHiGenEvtSelector {
 public:
   MultiCandGenEvtSelector(const edm::ParameterSet&);
   virtual ~MultiCandGenEvtSelector(){;}
   bool filter(HepMC::GenEvent *);

   double ptMin_;
   double etaMax_;
   int st_;
   int pdg_;
   int nTrig_;

};

#endif
