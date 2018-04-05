#ifndef __RecoParticleFlow_PFProducer_pfEGHelpersHeavyObjectCache_h__
#define __RecoParticleFlow_PFProducer_pfEGHelpersHeavyObjectCache_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include <memory>

namespace pfEGHelpers {
  class HeavyObjectCache {
  public:
    HeavyObjectCache(const edm::ParameterSet&);
    std::unique_ptr<const GBRForest> gbrEle_;
    std::unique_ptr<const GBRForest> gbrSingleLeg_;
  private:
    // for electron mva
    float lnPt_gsf, Eta_gsf, dPtOverPt_gsf, DPtOverPt_gsf, chi2_gsf, nhit_kf;
    float chi2_kf, EtotPinMode, EGsfPoutMode, EtotBremPinPoutMode, DEtaGsfEcalClust;
    float SigmaEtaEta, HOverHE, lateBrem, firstBrem;
    // for single leg mva
    float nlost, nlayers;
    float chi2, STIP, del_phi,HoverPt, EoverPt, track_pt;
  };
}

#endif // __RecoParticleFlow_PFProducer_pfEGHelpersHeavyObjectCache_h__
