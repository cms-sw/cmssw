import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.electrons_cff import *
from PhysicsTools.NanoAOD.electrons_cff import _eleVarsExtra
from PhysicsTools.NanoAOD.photons_cff import *
from PhysicsTools.NanoAOD.photons_cff import _phoVarsExtra
from PhysicsTools.NanoAOD.NanoAODEDMEventContent_cff import *
from PhysicsTools.NanoAOD.nanoDQM_cfi import nanoDQM
from PhysicsTools.NanoAOD.nanoDQM_cff import _Photon_extra_plots, _Electron_extra_plots

def addExtraEGammaVarsCustomize(process):
    #photon
    process.photonTable.variables.setValue(_phoVarsExtra.parameters_())
    if process.nanoDQM:
      process.nanoDQM.vplots.Photon.plots = _Photon_extra_plots
    #electron
    process.electronTable.variables.setValue(_eleVarsExtra.parameters_())
    if process.nanoDQM:
      process.nanoDQM.vplots.Electron.plots = _Electron_extra_plots
    return process
