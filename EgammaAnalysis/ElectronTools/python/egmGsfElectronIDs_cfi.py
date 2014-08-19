import FWCore.ParameterSet.Config as cms

egmGsfElectronIDs = cms.EDProducer(
    "VersionedElectronIdProducer",
    electronSrc = cms.InputTag('gedGsfElectrons'),
    electronsArePAT = cms.bool(False),
    electronIDs = cms.VPSet( )
)
    
