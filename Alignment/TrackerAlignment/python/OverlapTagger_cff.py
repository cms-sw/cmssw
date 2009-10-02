import FWCore.ParameterSet.Config as cms

OverlapTagger = cms.EDProducer("OverlapTagger",
                               src = cms.InputTag("generalTracks"),
                               Clustersrc = cms.InputTag("ALCARECOTkAlCosmicsCTF0T"),
                               rejectBadMods = cms.bool(False),
                               BadMods = cms.vuint32()

                               )###end of module
