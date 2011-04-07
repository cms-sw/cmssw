import FWCore.ParameterSet.Config as cms

## select events with high pfMET
pfMETSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("pfMet"),
    cut = cms.string( "pt()>150" )
    )

pfMETCounter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("pfMETSelector"),
    minNumber = cms.uint32(1),
    )

pfMETSelSeq = cms.Sequence(pfMETSelector*
                           pfMETCounter
                           )



## select events with high tcMET
tcMETSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("tcMet"),
    cut = cms.string( "pt()>500" )
    )

tcMETCounter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("tcMETSelector"),
    minNumber = cms.uint32(1),
    )

tcMETSelSeq = cms.Sequence(tcMETSelector*
                           tcMETCounter
                           )


