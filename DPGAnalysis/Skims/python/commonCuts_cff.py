import FWCore.ParameterSet.Config as cms

# ==========  CUT ON PRIMARY VERTEX

primaryVertexFilter = cms.EDFilter("GoodVertexFilter",
                                   vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                   minimumNDOF = cms.uint32(4) ,  #CHANGE FOR >= 3.5.4
                                   maxAbsZ = cms.double(15),
                                   maxd0 = cms.double(2)
                                   )

printoutModule = cms.EDAnalyzer("EventPrintout",
                                muonLabel = cms.InputTag("muons"),
                                photonLabel = cms.InputTag("photons"),
                                jetLabel = cms.InputTag("iterativeCone5CaloJets"),
                                electronLabel = cms.InputTag("gedGsfElectrons"),
                                triggerResults_ = cms.InputTag("TriggerResults","","HLT"),
                                ObjectMap = cms.InputTag("hltL1GtObjectMap"),
                                GtDigis = cms.InputTag("gtDigis")
                                )

plotsMakerModule = cms.EDAnalyzer("plotsMaker",
        HistOutFile = cms.untracked.string("l1Plots.root"),
        ObjectMap = cms.InputTag("hltL1GtObjectMap"),
        GtDigis = cms.InputTag("gtDigis"),
        defineBX = cms.untracked.int32(-1)
)

