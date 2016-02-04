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
  
  //void writeFile();
  
  virtual bool filter(edm::Event&, const edm::EventSetup&);
 private:

  const HepMC::GenEvent *myGenEvent;

  std::string label;
  double minptcut;
  double maxptcut;
  double minetacut;
  double maxetacut;
  int maxEvents;
  int nSelectedEvents, nGeneratedEvents;

  double ptSeedThr, etaSeedThr, ptGammaThr, etaGammaThr, ptTkThr, etaTkThr;
  double ptElThr, etaElThr, dRTkMax, dRSeedMax, dPhiSeedMax, dEtaSeedMax, dRNarrowCone, pTMinCandidate1, pTMinCandidate2, etaMaxCandidate;
  double invMassWide, invMassNarrow;
  int nTkConeMax, nTkConeSum;
  bool acceptPrompts;
  double promptPtThreshold;
  
  //std::string fileName;
  //TH1D *hPtSeed[2], *hEtaSeed[2], *hMassNarrow, *hMassWide;
  //TH1I *hPidSeed[2], *hNTk[2], *hSel, *hNTkSum;
  //TH1D *hPtCandidate[2], *hEtaCandidate[2];
  //TH1I *hPidCandidate[2];

};
#endif
