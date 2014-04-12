import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
allElectronTracks = cms.EDProducer("ChargedCandidateProducer",
    src = cms.InputTag("ctfWithMaterialTracks"),
    particleType = cms.string('e-')
)


