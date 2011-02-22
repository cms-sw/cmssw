import FWCore.ParameterSet.Config as cms

dtTTrigResidualCorrection = cms.EDAnalyzer("DTTTrigCorrection",
    dbLabel = cms.untracked.string(''),
    correctionAlgo = cms.string('DTTTrigResidualCorrection'),
    correctionAlgoConfig = cms.PSet(
        residualsRootFile = cms.string(''),
        #rootBaseDir = cms.untracked.string('/DQMData/DT/DTCalibValidation'),
        rootBaseDir = cms.untracked.string('DTResiduals'),
        useFitToResiduals = cms.bool(True)
    )
)
