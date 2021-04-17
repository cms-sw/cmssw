import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ppsAlignmentWorker = DQMEDAnalyzer("PPSAlignmentWorker",
	tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
	folder = cms.string("CalibPPS/Common"),
    label = cms.string(""),
	debug = cms.bool(False)
)