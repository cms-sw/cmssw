#ifndef PYTHIAHEPMCFILTERGAMMAGAMMA_h
#define PYTHIAHEPMCFILTERGAMMAGAMMA_h

//
// Package:    GeneratorInterface/GenFilters
// Class:      PythiaHepMCFilterGammaGamma
// 
// Original Author:  Matteo Sani
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "GeneratorInterface/Core/interface/BaseHepMCFilter.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
	  class HepMCProduct;
}

class PythiaHepMCFilterGammaGamma : public BaseHepMCFilter {
 public:
  explicit PythiaHepMCFilterGammaGamma(const edm::ParameterSet&);
  ~PythiaHepMCFilterGammaGamma() override;
  
  /** @return true if this GenEvent passes the double EM enrichment
      criterion */
  bool filter(const HepMC::GenEvent* myGenEvent) override;
 private:

  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  const int maxEvents;

  const double ptSeedThr, etaSeedThr, ptGammaThr, etaGammaThr, ptTkThr, etaTkThr;
  const double ptElThr, etaElThr, dRTkMax, dRSeedMax, dPhiSeedMax, dEtaSeedMax, dRNarrowCone, pTMinCandidate1, pTMinCandidate2, etaMaxCandidate;
  const double invMassMin, invMassMax;
  const double energyCut;
  const int nTkConeMax, nTkConeSum;
  const bool acceptPrompts;
  const double promptPtThreshold;

};
#endif
