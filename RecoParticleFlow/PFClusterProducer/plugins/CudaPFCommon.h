#ifndef CudaGenericPFlowPositionCalcCommon_h
#define CudaGenericPFlowPositionCalcCommon_h

#include <cmath>

namespace PFClustering {
  namespace common {
    enum PFLayer {
        PS2 = -12,
        PS1 = -11,
        ECAL_ENDCAP = -2,
        ECAL_BARREL = -1,
        NONE = 0,
        HCAL_BARREL1 = 1,
        HCAL_BARREL2 = 2,
        HCAL_ENDCAP = 3,
        HF_EM = 11,
        HF_HAD = 12,
        HGCAL = 13  // HGCal, could be EM or HAD
    };
    
    struct TimeResConsts {
        float corrTermLowE;
        float threshLowE;
        float noiseTerm;
        float constantTermLowE2;
        float noiseTermLowE;
        float threshHighE;
        float constantTerm2;
        float resHighE2;
        
        TimeResConsts() :
            corrTermLowE(-1.0),
            threshLowE(-1.0),
            noiseTerm(-1.0),
            constantTermLowE2(-1.0),
            noiseTermLowE(-1.0),
            threshHighE(-1.0),
            constantTerm2(-1.0),
            resHighE2(-1.0) {};

        TimeResConsts(float _corrTermLowE,
                      float _threshLowE,
                      float _noiseTerm,
                      float _constantTermLowE2,
                      float _noiseTermLowE,
                      float _threshHighE,
                      float _constantTerm2) :
            corrTermLowE(_corrTermLowE),
            threshLowE(_threshLowE),
            noiseTerm(_noiseTerm),
            constantTermLowE2(_constantTermLowE2),
            noiseTermLowE(_noiseTermLowE),
            threshHighE(_threshHighE),
            constantTerm2(_constantTerm2),
            resHighE2(std::pow((noiseTerm / threshHighE), 2) + constantTerm2) {};
    };

    struct PosCalcConfig {
        float minAllowedNormalization;
        float logWeightDenominatorInv;
        float minFractionInCalc;
        TimeResConsts timeResEndcap;
        TimeResConsts timeResBarrel;

        PosCalcConfig() :
            minAllowedNormalization(-1.0),
            logWeightDenominatorInv(-1.0),
            minFractionInCalc(-1.0) {};

        PosCalcConfig(float _minAllowedNormalization,
                      float _logWeightDenominatorInv,
                      float _minFractionInCalc,
                      TimeResConsts _timeResEndcap,
                      TimeResConsts _timeResBarrel) :
            minAllowedNormalization(_minAllowedNormalization),
            logWeightDenominatorInv(_logWeightDenominatorInv),
            minFractionInCalc(_minFractionInCalc),
            timeResEndcap(_timeResEndcap),
            timeResBarrel(_timeResBarrel) {};
    };

    
    // Config parameters for ECAL2DPositionCalcWithDepthCorr
    struct ECALPosDepthCalcConfig {
        float minAllowedNormalization;
        float T0_ES;
        float T0_EE;
        float T0_EB;
        float X0;
        float minFractionInCalc;
        float W0;

        ECALPosDepthCalcConfig() : 
            minAllowedNormalization(-1.0),
            T0_ES(-1.0),
            T0_EE(-1.0),
            T0_EB(-1.0),
            X0(-1.0),
            minFractionInCalc(-1.0),
            W0(-1.0) {};

        ECALPosDepthCalcConfig(float _minAllowedNormalization,
                               float _T0_ES,
                               float _T0_EE,
                               float _T0_EB,
                               float _X0,
                               float _minFractionInCalc,
                               float _W0) :
            minAllowedNormalization(_minAllowedNormalization),
            T0_ES(_T0_ES),
            T0_EE(_T0_EE),
            T0_EB(_T0_EB),
            X0(_X0),
            minFractionInCalc(_minFractionInCalc),
            W0(_W0) {};
    };
    
    struct CudaHCALConstants {
        float showerSigma2;
        float recHitEnergyNormInvEB_vec[4];
        float recHitEnergyNormInvEE_vec[7];
        float minFracToKeep;
        float minFracTot;
        float minFracInCalc;
        float minAllowedNormalization;
        uint32_t maxIterations;
        float stoppingTolerance;
        bool excludeOtherSeeds;
        float seedEThresholdEB_vec[4];
        float seedEThresholdEE_vec[7];
        float seedPt2ThresholdEB;
        float seedPt2ThresholdEE;
        float topoEThresholdEB_vec[4];
        float topoEThresholdEE_vec[7];
        int nNeigh;
        TimeResConsts endcapTimeResConsts;
        TimeResConsts barrelTimeResConsts;
    };
    
    struct CudaECALConstants {
        float showerSigma2;
        float recHitEnergyNormInvEB;
        float recHitEnergyNormInvEE;
        float minFracToKeep;
        float minFracTot;
        uint32_t maxIterations;
        float stoppingTolerance;
        bool excludeOtherSeeds;
        float seedEThresholdEB;
        float seedEThresholdEE;
        float seedPt2ThresholdEB;
        float seedPt2ThresholdEE;
        float topoEThresholdEB;
        float topoEThresholdEE;
        int nNeigh;
        PosCalcConfig posCalcConfig;
        ECALPosDepthCalcConfig convergencePosCalcConfig;
    };

  }  // namespace common
}   // namespace PFClustering


#endif // CudaGenericPFlowPositionCalcCommon_h

