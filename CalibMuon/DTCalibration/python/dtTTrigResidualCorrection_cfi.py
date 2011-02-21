import FWCore.ParameterSet.Config as cms

dtTTrigResidualCorrection = cms.EDAnalyzer("DTTTrigCorrection",
    dbLabel = cms.untracked.string(''),
    correctionAlgo = cms.string('DTTTrigResidualCorrection'),
    correctionAlgoConfig = cms.PSet(
        residualsRootFile = cms.string(''),
        useFitToResiduals = cms.bool(True)
    )
)
