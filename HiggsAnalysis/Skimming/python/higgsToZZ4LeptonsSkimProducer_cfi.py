import FWCore.ParameterSet.Config as cms

higgsToZZ4LeptonsSkimProducer = cms.EDFilter("HiggsToZZ4LeptonsSkimProducer",
    DebugHiggsToZZ4LeptonsSkim = cms.bool(False),

    # Collection to be accessed
    ElectronCollectionLabel = cms.InputTag("pixelMatchGsfElectrons"),
    MuonCollectionLabel     = cms.InputTag("muons"),
  
    # Pt thresholds for leptons
    stiffMinimumPt          = cms.double(10.0),
    softMinimumPt           = cms.double(5.0),

    # Minimum number of identified leptons above pt threshold
    nStiffLeptonMinimum     = cms.int32(2),

    # nLepton is nSoft + nStiff
    nLeptonMinimum          = cms.int32(3),
)


