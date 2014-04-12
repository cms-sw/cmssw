import FWCore.ParameterSet.Config as cms

from CalibMuon.RPCCalibration.l1Muon_EventContent_cff import *
L1MuonEventContent = cms.OutputModule("PoolOutputModule",
    l1Muon_EventContent,
    l1MuonEventSelection,
    datasets = cms.untracked.PSet(
        filterName = cms.untracked.string('l1Muon_Filter'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('l1Muon.root')
)


