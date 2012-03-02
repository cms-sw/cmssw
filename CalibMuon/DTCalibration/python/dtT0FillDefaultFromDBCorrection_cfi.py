import FWCore.ParameterSet.Config as cms

dtT0FillDefaultFromDBCorrection = cms.EDAnalyzer("DTT0Correction",
    correctionAlgo = cms.string('DTT0FillDefaultFromDB'),
    correctionAlgoConfig = cms.PSet(
        dbLabelRef = cms.string('t0Ref')
    )
)
