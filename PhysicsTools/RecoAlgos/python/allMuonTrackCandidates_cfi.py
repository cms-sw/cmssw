import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
allMuonTrackCandidates = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("ctfWithMaterialTracks"),
    particleType = cms.string('mu+')
)


# foo bar baz
# lK9KtAlShSmSI
# aj4mFfAozr6hy
