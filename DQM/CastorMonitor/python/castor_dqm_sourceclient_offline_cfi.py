import FWCore.ParameterSet.Config as cms

#--------------------------
# DQM Module
#--------------------------
# just put that for the time being

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
castorOfflineMonitor = DQMEDAnalyzer('CastorMonitorModule',
   debug = cms.untracked.int32(0), #(=0 - no messages)
                           # Turn on/off timing diagnostic
                           showTiming          = cms.untracked.bool(False),

     l1tStage2uGtSource = cms.InputTag("gtStage2Digis"),
     tagTriggerResults   = cms.InputTag('TriggerResults','','HLT'),
    HltPaths  = cms.vstring("HLT_ZeroBias","HLT_Random"),

                           digiLabel            = cms.InputTag("castorDigis"),
                           rawLabel             = cms.InputTag("rawDataCollector"),
                           unpackerReportLabel  = cms.InputTag("castorDigis"),
                           CastorRecHitLabel    = cms.InputTag("castorreco"),
                           CastorTowerLabel     = cms.InputTag("CastorTowerReco"),
                           CastorBasicJetsLabel = cms.InputTag("ak7CastorJets"),
                           CastorJetIDLabel     = cms.InputTag("ak7CastorJetID"),

                           DataIntMonitor= cms.untracked.bool(True),
                           TowerJetMonitor= cms.untracked.bool(True),

                           DigiMonitor = cms.untracked.bool(True),

                           RecHitMonitor = cms.untracked.bool(True),


#                           LEDMonitor = cms.untracked.bool(True),
#                           LEDPerChannel = cms.untracked.bool(True),
                           FirstSignalBin = cms.untracked.int32(0),
                           LastSignalBin = cms.untracked.int32(9)
)

