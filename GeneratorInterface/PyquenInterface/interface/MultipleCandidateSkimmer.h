#ifndef _HI_DiMuonSkimmer_h__
#define _HI_DiMuonSkimmer_h__

#include "GeneratorInterface/PyquenInterface/interface/BaseHiGenSkimmer.h"

class MultipleCandidateSkimmer : public BaseHiGenSkimmer {
 public:
   MultipleCandidateSkimmer(const edm::ParameterSet&);
   virtual ~MultipleCandidateSkimmer(){;}
   bool filter(HepMC::GenEvent *);

   double ptMin_;
   double etaMax_;
   int st_;
   int pdg_;
   int nTrig_;

};

#endif
