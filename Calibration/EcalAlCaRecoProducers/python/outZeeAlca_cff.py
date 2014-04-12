import FWCore.ParameterSet.Config as cms

from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalElectron_Output_cff import *
outZeeAlca = cms.OutputModule("PoolOutputModule",
    OutALCARECOEcalCalElectron,
    fileName = cms.untracked.string('alCaElectrons_se.root')
)


