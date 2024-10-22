#ifndef RecoParticleFlow_PFClusterProducer_interface_PFClusterParamsSoA_h
#define RecoParticleFlow_PFClusterProducer_interface_PFClusterParamsSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  GENERATE_SOA_LAYOUT(PFClusterParamsSoALayout,
                      SOA_SCALAR(int32_t, nNeigh),
                      SOA_SCALAR(float, seedPt2ThresholdHB),
                      SOA_SCALAR(float, seedPt2ThresholdHE),
                      SOA_COLUMN(float, seedEThresholdHB_vec),
                      SOA_COLUMN(float, seedEThresholdHE_vec),
                      SOA_COLUMN(float, topoEThresholdHB_vec),
                      SOA_COLUMN(float, topoEThresholdHE_vec),
                      SOA_SCALAR(float, showerSigma2),
                      SOA_SCALAR(float, minFracToKeep),
                      SOA_SCALAR(float, minFracTot),
                      SOA_SCALAR(uint32_t, maxIterations),
                      SOA_SCALAR(bool, excludeOtherSeeds),
                      SOA_SCALAR(float, stoppingTolerance),
                      SOA_SCALAR(float, minFracInCalc),
                      SOA_SCALAR(float, minAllowedNormalization),
                      SOA_COLUMN(float, recHitEnergyNormInvHB_vec),
                      SOA_COLUMN(float, recHitEnergyNormInvHE_vec),
                      SOA_SCALAR(float, barrelTimeResConsts_corrTermLowE),
                      SOA_SCALAR(float, barrelTimeResConsts_threshLowE),
                      SOA_SCALAR(float, barrelTimeResConsts_noiseTerm),
                      SOA_SCALAR(float, barrelTimeResConsts_constantTermLowE2),
                      SOA_SCALAR(float, barrelTimeResConsts_noiseTermLowE),
                      SOA_SCALAR(float, barrelTimeResConsts_threshHighE),
                      SOA_SCALAR(float, barrelTimeResConsts_constantTerm2),
                      SOA_SCALAR(float, barrelTimeResConsts_resHighE2),
                      SOA_SCALAR(float, endcapTimeResConsts_corrTermLowE),
                      SOA_SCALAR(float, endcapTimeResConsts_threshLowE),
                      SOA_SCALAR(float, endcapTimeResConsts_noiseTerm),
                      SOA_SCALAR(float, endcapTimeResConsts_constantTermLowE2),
                      SOA_SCALAR(float, endcapTimeResConsts_noiseTermLowE),
                      SOA_SCALAR(float, endcapTimeResConsts_threshHighE),
                      SOA_SCALAR(float, endcapTimeResConsts_constantTerm2),
                      SOA_SCALAR(float, endcapTimeResConsts_resHighE2))

  using PFClusterParamsSoA = PFClusterParamsSoALayout<>;

}  // namespace reco

#endif
