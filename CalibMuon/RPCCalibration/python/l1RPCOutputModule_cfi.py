import FWCore.ParameterSet.Config as cms

from CalibMuon.RPCCalibration.l1RPC_EventContent_cff import *
L1RPCEventContent = cms.OutputModule("PoolOutputModule",
    l1RPCEventSelection,
    l1RPC_EventContent,
    datasets = cms.untracked.PSet(
        filterName = cms.untracked.string('l1RPC_Filter'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('l1RPC.root')
)


