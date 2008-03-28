import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.photonjets_EventContent_cff import *
from JetMETAnalysis.JetSkims.FEVTSIMPhotonJetsEventContent_cff import *
photonjetsOutputModuleFEVTSIM = cms.OutputModule("PoolOutputModule",
    FEVTSIMPhotonJetsEventContent,
    photonjetsEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('photonjets_FEVTSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('photonjets_FEVTSIM.root')
)


