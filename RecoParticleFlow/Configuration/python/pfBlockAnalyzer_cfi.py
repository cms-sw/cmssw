import FWCore.ParameterSet.Config as cms

pfBlockAnalyzer = cms.EDAnalyzer("BlockAnalyzer",
                                 blockCollection = cms.InputTag("particleFlowBlock"),
                                 trackCollection = cms.InputTag("generalTracks"),
                                 PrimaryVertexLabel = cms.InputTag("offlinePrimaryVertices"),
                                 OutputFile = cms.string('dummy.root')
                                 )
