import FWCore.ParameterSet.Config as cms

gedPhotonCoreGSFixed = cms.EDProducer("GEDPhotonCoreGSCrysFixer",
    photonCores = cms.InputTag("gedPhotonCore", '', cms.InputTag.skipCurrentProcess()),
    refinedSCs = cms.InputTag('particleFlowEGammaGSFixed'),
    scs = cms.InputTag('particleFlowSuperClusterECALGSFixed'),
    conversions = cms.InputTag('allConversions'),
    singleconversions = cms.InputTag('particleFlowEGamma')
)
