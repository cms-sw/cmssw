import FWCore.ParameterSet.Config as cms


## select events with at least one good PV
pvFilter = cms.EDFilter(
    "VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
    filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
    )


## apply HBHE Noise filter
import CommonTools.RecoAlgos.HBHENoiseFilter_cfi
HBHENoiseFilter = CommonTools.RecoAlgos.HBHENoiseFilter_cfi.HBHENoiseFilter.clone()


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

pfMETSelSeq = cms.Sequence(pvFilter*
                           HBHENoiseFilter*
                           pfMETSelector*
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

tcMETSelSeq = cms.Sequence(pvFilter*
                           HBHENoiseFilter*
                           tcMETSelector*
                           tcMETCounter
                           )


