import FWCore.ParameterSet.Config as cms

dtT0WireInChamberReferenceCorrection = cms.EDAnalyzer("DTT0Correction",
    correctionAlgo = cms.string('DTT0WireInChamberReferenceCorrection'),
    correctionAlgoConfig = cms.PSet(
        # Format "wheel station sector" (e.g. "-1 3 10")
        calibChamber = cms.string('All') 
    )
)
