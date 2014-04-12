import FWCore.ParameterSet.Config as cms

L1HLTJetsMatching= cms.EDProducer( "L1HLTTauMatching",
    JetSrc = cms.InputTag( 'hltL2TauRelaxingIsolationSelector','Isolated' ),
    L1TauTrigger = cms.InputTag( "hltL1sDoubleLooseIsoTau15" ),
    EtMin = cms.double( 15.0 )
)

