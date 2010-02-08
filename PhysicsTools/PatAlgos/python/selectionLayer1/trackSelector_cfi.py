import FWCore.ParameterSet.Config as cms

# module to select Taus
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
selectedLayer1TrackCands = cms.EDFilter("PATGenericParticleSelector",
    src = cms.InputTag("allLayer1TrackCands"),
    cut = cms.string("")
)


