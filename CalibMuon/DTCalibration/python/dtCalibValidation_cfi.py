import FWCore.ParameterSet.Config as cms

dtCalibValidation = DQMStep1Module('DTCalibValidation',
    # Write the histos on file
    OutputMEsInRootFile = cms.bool(False),
    # Lable to retrieve 2D segments from the event
    segment2DLabel = cms.untracked.string('dt2DSegments'),
    OutputFileName = cms.string('residuals.root'),
    # Lable to retrieve 4D segments from the event
    segment4DLabel = cms.untracked.string('dt4DSegments'),
    debug = cms.untracked.bool(False),
    # Lable to retrieve RecHits from the event
    recHits1DLabel = cms.untracked.string('dt1DRecHits'),
    # Detailed analysis
    detailedAnalysis = cms.untracked.bool(False)
)
