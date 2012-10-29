#ifndef _HI_EcalGenEvtSelectorFrag_h__
#define _HI_EcalGenEvtSelectorFrag_h__

#include <vector>
#include "GeneratorInterface/HiGenCommon/interface/BaseHiGenEvtSelector.h"

class EcalGenEvtSelectorFrag : public BaseHiGenEvtSelector {
 public:
   EcalGenEvtSelectorFrag(const edm::ParameterSet& pset);
   virtual ~EcalGenEvtSelectorFrag(){;}

   bool filter(HepMC::GenEvent *);

 private:

   std::vector<int> partonId_;
   std::vector<int> partonStatus_;
   std::vector<double> partonPt_;

   std::vector<int> particleId_;
   std::vector<int> particleStatus_;
   std::vector<double> particlePt_;

   double etaMax_;
};

#endif
