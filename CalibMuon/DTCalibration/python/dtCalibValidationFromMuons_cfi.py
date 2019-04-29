import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from RecoMuon.TrackingTools.MuonSegmentMatcher_cff import *

dtCalibValidation = DQMEDAnalyzer("DTCalibValidationFromMuons",
    MuonSegmentMatcher,
    OutputFileName = cms.string('residuals.root'),
    # Write the histos on file
    OutputMEsInRootFile = cms.bool(False),
    debug = cms.untracked.bool(False),
    # Detailed analysis
    detailedAnalysis = cms.untracked.bool(False),
    # Muon collection used for matching
    muonLabel = cms.untracked.string('muons'),
    # Lable to retrieve RecHits from the event
    recHits1DLabel = cms.untracked.string('dt1DRecHits'),
    # Lable to retrieve 2D segments from the event
    segment2DLabel = cms.untracked.string('dt2DSegments'),
    # Lable to retrieve 4D segments from the event
    segment4DLabel = cms.untracked.string('dt4DSegments')
)

