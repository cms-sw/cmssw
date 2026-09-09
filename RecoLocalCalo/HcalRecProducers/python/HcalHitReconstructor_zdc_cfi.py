import FWCore.ParameterSet.Config as cms

zdcreco = cms.EDProducer(
    "ZdcHitReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabelhcal = cms.InputTag("hcalDigis"),
    digiLabelcastor = cms.InputTag("castorDigis"),
    digiLabelQIE10ZDC = cms.InputTag("hcalDigis:ZDC"),
    Subdetector = cms.string('ZDC'),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False),
    dropZSmarkedPassed = cms.bool(True),
    recoMethod = cms.int32(2),
    lowGainOffset = cms.int32(1),
    lowGainFrac = cms.double(8.15),

    # Set Time Samples of all digis to be saved in aux word
    # ZDC would like the ability to store non-contiguous digis
    AuxTSvec = cms.vint32([4,5,6,7]),
        
    #Tags for calculating status flags
    # None of the flag algorithms have been implemented for zdc, so these booleans do nothing
    correctTiming = cms.bool(True),
    setNoiseFlags = cms.bool(True),
    setHSCPFlags  = cms.bool(True),
    setSaturationFlags = cms.bool(True),
    setTimingTrustFlags = cms.bool(False), # timing flags currently only implemented for HF
    
    saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127))
    ) # zdcreco

from Configuration.Eras.Modifier_run3_common_cff import run3_common
from RecoLocalCalo.HcalRecProducers.zdcrecoRun3_cfi import zdcrecoRun3 as _zdcrecoRun3
run3_common.toReplaceWith(zdcreco, _zdcrecoRun3.clone())
