import FWCore.ParameterSet.Config as cms

#==============================================================================
# Sequence to the GED electrons.
#==============================================================================

from RecoEgamma.EgammaElectronProducers.gedGsfElectronCores_cfi import *
from RecoEgamma.EgammaElectronProducers.gedGsfElectrons_cfi import *
from RecoEgamma.EgammaElectronProducers.gedGsfElectronValueMapsTmp_cfi import *

gedGsfElectronTaskTmp = cms.Task(gedGsfElectronCores, gedGsfElectronsTmp, gedGsfElectronValueMapsTmp)

# in Run 3 the electron and photon PFID DNNs use the ONNX Runtime, which requires the ONNXService
from Configuration.Eras.Modifier_run3_common_cff import run3_common

def _addONNXService(process):
    process.load("PhysicsTools.ONNXRuntime.ONNXService_cfi")

modifyGedGsfElectronSequence_addONNXService = run3_common.makeProcessModifier(_addONNXService)
