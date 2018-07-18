import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
FSQDQM= DQMEDAnalyzer('FSQDQM',
                      triggerResultsCollection = cms.InputTag("TriggerResults", "", "HLT"),
                      HLTriggerResults = cms.InputTag("TriggerResults","","HLT"),
                      LabelPFJet       = cms.string("ak4PFJetsCHS"),
                      LabelCastorJet   = cms.string("ak5CastorJets"),
                      LabelTrack       = cms.string("generalTracks"),
                      LabelBeamSpot    = cms.string("offlineBeamSpot"),
                      #                      LabelVertex      = cms.string("offlinePrimaryVertices"),
                      pvs = cms.InputTag('offlinePrimaryVertices'),                      
                      )
