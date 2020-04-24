import FWCore.ParameterSet.Config as cms

PUDumper = cms.EDAnalyzer('PUDumper',
                          pileupSummary = cms.InputTag("addPileupInfo"),
#                               vertexCollection = cms.InputTag('offlinePrimaryVertices'),
#                               foutName = cms.string("ZShervinNtuple.root"),
                          )
