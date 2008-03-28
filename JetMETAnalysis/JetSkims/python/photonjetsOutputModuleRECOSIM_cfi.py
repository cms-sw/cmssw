import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.photonjets_EventContent_cff import *
from JetMETAnalysis.JetSkims.RECOSIMPhotonJetsEventContent_cff import *
photonjetsOutputModuleRECOSIM = cms.OutputModule("PoolOutputModule",
    photonjetsEventSelection,
    RECOSIMPhotonJetsEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('photonjets_RECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('photonjets_RECOSIM.root')
)


