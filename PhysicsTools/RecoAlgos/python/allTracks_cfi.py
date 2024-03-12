import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
allTracks = cms.EDProducer("ChargedCandidateProducer",
    src = cms.InputTag("generalTracks"),
    particleType = cms.string('pi+')
)


# foo bar baz
# z0CFMiI7omVuB
# bsfzPCXKlQf0c
