import FWCore.ParameterSet.Config as cms

dtT0FillChamberFromDBCorrection = cms.EDAnalyzer("DTT0Correction",
    correctionAlgo = cms.string('DTT0FillChamberFromDB'),
    correctionAlgoConfig = cms.PSet(
        dbLabelRef = cms.string('t0Ref'),
        # Format "wheel station sector" (e.g. "-1 3 10")
        chamberId = cms.string('') 
    )
)
