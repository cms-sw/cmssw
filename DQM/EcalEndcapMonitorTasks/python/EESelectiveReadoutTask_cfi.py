import FWCore.ParameterSet.Config as cms

ecalEndcapSelectiveReadoutTask = cms.EDAnalyzer("EESelectiveReadoutTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    EEDigiCollection = cms.InputTag("ecalEBunpacker:eeDigis"),
    EEUsuppressedDigiCollection = cms.InputTag("ecalUnsuppressedDigis"),
    EESRFlagCollection = cms.InputTag("ecalEBunpacker"),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalEBunpacker:EcalTriggerPrimitives"),
    FEDRawDataCollection = cms.InputTag("rawDataCollector"),
    configFromCondDB = cms.bool(False),
    # if configFromCondDB is true, dccWeights are not used.
    dccWeights = cms.vdouble(-0.374, -0.374, -0.3629, 0.2721, 0.4681, 0.3707),
    ecalDccZs1stSample = cms.int32(2)
)
