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
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "TH1D.h"
#include "TH1I.h"

class PythiaFilterGammaGamma : public edm::EDFilter {
 public:
  explicit PythiaFilterGammaGamma(const edm::ParameterSet&);
  ~PythiaFilterGammaGamma();
  
  virtual bool filter(edm::Event&, const edm::EventSetup&);
 private:

  const HepMC::GenEvent *myGenEvent;

  edm::EDGetTokenT<edm::HepMCProduct> token_;
  double minptcut;
  double maxptcut;
  double minetacut;
  double maxetacut;
  int maxEvents;
  int nSelectedEvents, nGeneratedEvents, counterPrompt;

  double ptSeedThr, etaSeedThr, ptGammaThr, etaGammaThr, ptTkThr, etaTkThr;
  double ptElThr, etaElThr, dRTkMax, dRSeedMax, dPhiSeedMax, dEtaSeedMax, dRNarrowCone, pTMinCandidate1, pTMinCandidate2, etaMaxCandidate;
  double invMassMin, invMassMax;
  double energyCut;
  int nTkConeMax, nTkConeSum;
  bool acceptPrompts;
  double promptPtThreshold;

};
#endif
