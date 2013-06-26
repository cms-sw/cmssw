#ifndef _HI_EcalGenEvtSelector_h__
#define _HI_EcalGenEvtSelector_h__

#include <vector>
#include "GeneratorInterface/HiGenCommon/interface/BaseHiGenEvtSelector.h"

class EcalGenEvtSelector : public BaseHiGenEvtSelector {
 public:
   EcalGenEvtSelector(const edm::ParameterSet& pset);
   virtual ~EcalGenEvtSelector(){;}

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
