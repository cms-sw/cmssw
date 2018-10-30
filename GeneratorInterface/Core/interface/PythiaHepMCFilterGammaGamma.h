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

  //----------
  // filter parameters
  //----------

  /** minimum pt and maximum absolute eta for electron and photon seeds */
  const double ptSeedThr, etaSeedThr;

  /** minimum pt and maximum absolute eta for photons to be added
      to seeds to form candidates (see also ptElThr, etaElThr) */
  const double ptGammaThr, etaGammaThr;

  /** minimum pt and maximum absolute eta for charged (stable) particles
      to be counted in the isolation cone */
  const double ptTkThr, etaTkThr;

  /** minimum pt and maximum absolute eta for electrons to be added
      to seeds to form candidates (see also ptGammaThr, etaGammaThr) */
  const double ptElThr, etaElThr;

  /** delta R of cone around candidates in which charged tracks are counted */
  const double dRTkMax;

  /** delta R of cone around seeds in which other electrons/photons are
      added to seeds to form candidates */
  const double dRSeedMax;

  /** maximum difference in phi and eta for which other electrons/photons
      are added to seeds to form candidates.

      Note that electrons/photons are accepted if they are within the cone
      specified by dRSeedMax or if they are within the rectangular region
      specified by (dPhiSeedMax, dEtaSeedMax). */
  const double dPhiSeedMax, dEtaSeedMax;

  /** this parameter is effectively unused */
  const double dRNarrowCone;

  /** minimum pt for leading and subleading candidate */
  const double pTMinCandidate1, pTMinCandidate2;

  /** maximum absolute eta for candidates */
  const double etaMaxCandidate;

  /** invariant mass range for mass of a pair of candidates */
  const double invMassMin, invMassMax;

  /** minimum energy for both candidates */
  const double energyCut;

  /** maximum number of charged particles in the isolation cone
      around each candidate */
  const int nTkConeMax;

  /** maximum number of charged particles summed over both
      cones of a pair of candidates */
  const int nTkConeSum;

  /** if true, prompt seeds (electrons/photons with no mother
      or only ancestors of the same type) will be considered
      as having zero charged tracks in the isolation cone */
  const bool acceptPrompts;

  /** minimum pt for prompt seed particles to be considered (only
      effective if acceptPrompts is true) */
  const double promptPtThreshold;

};
#endif
