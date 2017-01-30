import FWCore.ParameterSet.Config as cms

gedGsfElectronCoresGSFixed = cms.EDProducer("GsfElectronCoreGSCrysFixer",
    orgCores=cms.InputTag("gedGsfElectronCores", '', cms.InputTag.skipCurrentProcess()),
    refinedSCs = cms.InputTag('particleFlowEGammaGSFixed'),
    scs = cms.InputTag('particleFlowSuperClusterECALGSFixed'),
)
