import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.photonjets_EventContent_cff import *
from JetMETAnalysis.JetSkims.AODSIMPhotonJetsEventContent_cff import *
photonjetsOutputModuleAODSIM = cms.OutputModule("PoolOutputModule",
    photonjetsEventSelection,
    AODSIMPhotonJetsEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('photonjets_AODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('photonjets_AODSIM.root')
)


