import FWCore.ParameterSet.Config as cms

OverlapTagger = cms.EDProducer("OverlapTagger",
                               src = cms.InputTag("generalTracks"),
                               Clustersrc = cms.InputTag("ALCARECOTkAlCosmicsCTF0T"),
                               rejectBadMods = cms.bool(False),
                               BadMods = cms.vuint32()

                               )###end of module
# foo bar baz
# e9bOIpq0qWDE6
# Ri49l6N8Kk5HD
