import FWCore.ParameterSet.Config as cms

#ctppsPixelDQMSource = cms.EDAnalyzer("CTPPSPixelDQMSource",
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ctppsPixelDQMSource = DQMEDAnalyzer('CTPPSPixelDQMSource',
    tagRPixDigi = cms.untracked.InputTag("ctppsPixelDigis", ""),
    tagRPixError = cms.untracked.InputTag("ctppsPixelDigis", ""),
    tagRPixCluster = cms.untracked.InputTag("ctppsPixelClusters", ""),  
    tagRPixLTrack = cms.untracked.InputTag("ctppsPixelLocalTracks", ""),  
    tagTrigResults = cms.untracked.InputTag("TriggerResults","","HLT"),
    RPStatusWord = cms.untracked.uint32(0x8008), # rpots in readout:220_fr_hr; 210_fr_hr
    verbosity = cms.untracked.uint32(0),
    randomHLTPath = cms.untracked.string("HLT_Random_v3"),
    offlinePlots = cms.untracked.bool(False),
    onlinePlots = cms.untracked.bool(True),
    turnOffPlanePlots = cms.untracked.vstring(), 	# add tags for planes to be shut off, 
    												# e.g. "0_2_3_4" for arm 0 station 2 rp 3 plane 4
)

ctppsPixelDQMOfflineSource = DQMEDAnalyzer('CTPPSPixelDQMSource',
    tagRPixDigi = cms.untracked.InputTag("ctppsPixelDigis", ""),
    tagRPixError = cms.untracked.InputTag("ctppsPixelDigis", ""),
    tagRPixCluster = cms.untracked.InputTag("ctppsPixelClusters", ""),  
    tagRPixLTrack = cms.untracked.InputTag("ctppsPixelLocalTracks", ""),  
    tagTrigResults = cms.untracked.InputTag("TriggerResults","","HLT"),
    RPStatusWord = cms.untracked.uint32(0x8008), # rpots in readout: 220_fr_hr; 210_fr_hr
    verbosity = cms.untracked.uint32(0),
    randomHLTPath = cms.untracked.string("HLT_Random_v3"),
    offlinePlots = cms.untracked.bool(True),
    onlinePlots = cms.untracked.bool(False),
    turnOffPlanePlots = cms.untracked.vstring(), 	# add tags for planes to be shut off, 
    												# e.g. "0_2_3_4" for arm 0 station 2 rp 3 plane 4
)
