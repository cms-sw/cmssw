import FWCore.ParameterSet.Config as cms

# module to select Taus
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
selectedPatTracks = cms.EDFilter("PATGenericParticleSelector",
    src = cms.InputTag("allPatTracks"),
    cut = cms.string("")
)


