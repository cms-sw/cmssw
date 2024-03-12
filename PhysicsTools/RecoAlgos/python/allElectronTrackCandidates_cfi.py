import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
allElectronTrackCandidates = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("ctfWithMaterialTracks"),
    particleType = cms.string('e-')
)


# foo bar baz
# 2Msrr7nH8cfTL
