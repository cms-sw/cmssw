#ifndef _HI_EcalCandSkimmer_h__
#define _HI_EcalCandSkimmer_h__

#include <vector>
#include "GeneratorInterface/PyquenInterface/interface/BaseHiGenSkimmer.h"

class EcalCandidateSkimmer : public BaseHiGenSkimmer {
 public:
   EcalCandidateSkimmer(const edm::ParameterSet& pset);
   virtual ~EcalCandidateSkimmer(){;}

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
