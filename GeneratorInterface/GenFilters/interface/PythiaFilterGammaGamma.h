#ifndef PYTHIAFILTERGAMMAGAMMA_h
#define PYTHIAFILTERGAMMAGAMMA_h

//
// Package:    GeneratorInterface/GenFilters
// Class:      PythiaFilterGammaGamma
// 
// Original Author:  Matteo Sani
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
	  class HepMCProduct;
}

class PythiaFilterGammaGamma : public edm::global::EDFilter<> {
 public:
  explicit PythiaFilterGammaGamma(const edm::ParameterSet&);
  ~PythiaFilterGammaGamma() override;
  
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
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
