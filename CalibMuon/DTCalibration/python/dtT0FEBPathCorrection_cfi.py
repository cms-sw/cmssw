import FWCore.ParameterSet.Config as cms

dtT0FEBPathCorrection = cms.EDAnalyzer("DTT0Correction",
    correctionAlgo = cms.string('DTT0FEBPathCorrection'),
    correctionAlgoConfig = cms.PSet(
        # Format "wheel station sector" (e.g. "-1 3 10")
        calibChamber = cms.string('All'),
    )
)
