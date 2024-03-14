import FWCore.ParameterSet.Config as cms

OverlapTagger = cms.EDProducer("TkAlCaOverlapTagger",
                               src = cms.InputTag("generalTracks"),
                               Clustersrc = cms.InputTag("ALCARECOTkAlCosmicsCTF0T"),
                               rejectBadMods = cms.bool(False),
                               BadMods = cms.vuint32()

                               )###end of module
# dummy dummy
