import FWCore.ParameterSet.Config as cms

hfreco = cms.EDFilter(
    "HcalHitReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabel = cms.InputTag("hcalZeroSuppressedDigis"),
    
    samplesToAdd = cms.int32(1),
    Subdetector = cms.string('HF'),
    firstSample = cms.int32(3),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False),
    
    # Tags for calculating status flags
    correctTiming = cms.bool(True),
    setNoiseFlags = cms.bool(True),
    setHSCPFlags  = cms.bool(True), # HSCP flags not implemented for hfreco; this boolean does nothing
    setSaturationFlags = cms.bool(True),
    
    digistat= cms.PSet(
    HFpulsetimemin     = cms.int32(0),
    HFpulsetimemax     = cms.int32(10), # min/max time slice values for peak
    HFratio_beforepeak = cms.double(0.1), # max allowed ratio
    HFratio_afterpeak  = cms.double(1.0), # max allowed ratio
    adcthreshold       = cms.int32(10), # minimum size of peak (in ADC counts, after ped subtraction) to be considered noisy
    ),
    rechitstat=cms.PSet(
    HFlongshortratio = cms.double(0.99), # max allowed ratio of (L-S)/(L+S)
    HFthresholdET = cms.double(0.50), # minimum energy (in GeV) required for a cell to be considered hot
    ),
    saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127))                      
    ) # hfreco


