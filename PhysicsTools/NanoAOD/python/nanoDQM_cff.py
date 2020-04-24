import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.nanoDQM_cfi import nanoDQM
from PhysicsTools.NanoAOD.nanoDQM_tools_cff import *

## Modify plots accordingly to era
from Configuration.StandardSequences.Eras import eras
_vplots80X = nanoDQM.vplots.clone()
# Tau plots
_tauPlots80X = cms.VPSet()
for plot in _vplots80X.Tau.plots:
    if plot.name.value().find("MVA")>-1 and plot.name.value().find("2017")>-1:
        continue
    _tauPlots80X.append(plot)
_tauPlots80X.append(Plot1D('idMVAnewDM', 'idMVAnewDM', 64, -0.5, 63.5, 'IsolationMVArun2v1DBnewDMwLT ID working point: bitmask 1 = VLoose, 2 = Loose, 4 = Medium, 8 = Tight, 16 = VTight, 32 = VVTight'))
_tauPlots80X.append(Plot1D('idMVAoldDMdR03', 'idMVAoldDMdR03', 64, -0.5, 63.5, 'IsolationMVArun2v1DBdR03oldDMwLT ID working point: bitmask 1 = VLoose, 2 = Loose, 4 = Medium, 8 = Tight, 16 = VTight, 32 = VVTight'))
_tauPlots80X.append(Plot1D('rawMVAnewDM', 'rawMVAnewDM', 20, -1, 1, 'byIsolationMVArun2v1DBnewDMwLT raw output discriminator'))
_tauPlots80X.append(Plot1D('rawMVAoldDMdR03', 'rawMVAoldDMdR03', 20, -1, 1, 'byIsolationMVArun2v1DBdR03oldDMwLT raw output discriminator'))
_vplots80X.Tau.plots = _tauPlots80X
eras.run2_miniAOD_80XLegacy.toModify(nanoDQM,
                                     vplots = _vplots80X
)

## MC
nanoDQMMC = nanoDQM.clone()
nanoDQMMC.vplots.Electron.sels.Prompt = cms.string("genPartFlav == 1")
nanoDQMMC.vplots.Muon.sels.Prompt = cms.string("genPartFlav == 1")
nanoDQMMC.vplots.Photon.sels.Prompt = cms.string("genPartFlav == 1")
nanoDQMMC.vplots.Tau.sels.Prompt = cms.string("genPartFlav == 5")
nanoDQMMC.vplots.Jet.sels.Prompt = cms.string("genJetIdx != 1")
nanoDQMMC.vplots.Jet.sels.PromptB = cms.string("genJetIdx != 1 && hadronFlavour == 5")

nanoDQMQTester = cms.EDAnalyzer("QualityTester",
    qtList = cms.untracked.FileInPath('PhysicsTools/NanoAOD/test/dqmQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    testInEventloop = cms.untracked.bool(False),
    qtestOnEndLumi = cms.untracked.bool(False),
    verboseQT =  cms.untracked.bool(True)
)

nanoHarvest = cms.Sequence( nanoDQMQTester )
