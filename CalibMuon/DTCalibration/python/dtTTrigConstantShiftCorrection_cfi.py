import FWCore.ParameterSet.Config as cms

dtTTrigConstantShiftCorrection = cms.EDAnalyzer("DTTTrigCorrection",
    dbLabel = cms.untracked.string(''),
    correctionAlgo = cms.string('DTTTrigConstantShift'),
    correctionAlgoConfig = cms.PSet(
        dbLabel = cms.untracked.string(''),
        value = cms.double(0.), 
        # Format "wheel station sector" (e.g. "-1 3 10")
        calibChamber = cms.string('All')
    )
)
