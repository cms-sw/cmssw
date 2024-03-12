import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
allMuonTracks = cms.EDProducer("ChargedCandidateProducer",
    src = cms.InputTag("ctfWithMaterialTracks"),
    particleType = cms.string('mu+')
)


# foo bar baz
# Nf35vhThB6nq5
# qiYYOfPsD3vAi
