import FWCore.ParameterSet.Config as cms

#ctppsPixelDQMSource = cms.EDAnalyzer("CTPPSPixelDQMSource",
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ctppsPixelDQMSource = DQMEDAnalyzer('CTPPSPixelDQMSource',
    tagRPixDigi = cms.InputTag("ctppsPixelDigis", ""),
    tagRPixCluster = cms.InputTag("ctppsPixelClusters", ""),  
    tagRPixLTrack = cms.InputTag("ctppsPixelLocalTracks", ""),  
    RPStatusWord = cms.untracked.uint32(0x8008), # rpots in readout:220_fr_hr; 210_fr_hr
    verbosity = cms.untracked.uint32(0),
    offlinePlots = cms.untracked.bool(False),
    onlinePlots = cms.untracked.bool(True),
    turnOffPlanePlots = cms.untracked.vstring(), 	# add tags for planes to be shut off, 
    												# e.g. "0_2_3_4" for arm 0 station 2 rp 3 plane 4
)

ctppsPixelDQMOfflineSource = DQMEDAnalyzer('CTPPSPixelDQMSource',
    tagRPixDigi = cms.InputTag("ctppsPixelDigis", ""),
    tagRPixCluster = cms.InputTag("ctppsPixelClusters", ""),  
    tagRPixLTrack = cms.InputTag("ctppsPixelLocalTracks", ""),  
    RPStatusWord = cms.untracked.uint32(0x8008), # rpots in readout: 220_fr_hr; 210_fr_hr
    verbosity = cms.untracked.uint32(0),
    offlinePlots = cms.untracked.bool(True),
    onlinePlots = cms.untracked.bool(False),
    turnOffPlanePlots = cms.untracked.vstring(), 	# add tags for planes to be shut off, 
    												# e.g. "0_2_3_4" for arm 0 station 2 rp 3 plane 4
)