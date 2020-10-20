import FWCore.ParameterSet.Config as cms
import copy

from PhysicsTools.NanoAOD.nanoDQM_cfi import nanoDQM
from PhysicsTools.NanoAOD.nanoDQM_tools_cff import *
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

nanogenDQM = DQMEDAnalyzer("NanoAODDQM",
    vplots = cms.PSet(GenDressedLepton = nanoDQM.vplots.GenDressedLepton,
        GenIsolatedPhoton = nanoDQM.vplots.GenIsolatedPhoton,
        GenJet = nanoDQM.vplots.GenJet,
        GenJetAK8 = nanoDQM.vplots.GenJetAK8,
        GenMET = nanoDQM.vplots.GenMET,
        GenPart = nanoDQM.vplots.GenPart, 
        GenVisTau = nanoDQM.vplots.GenVisTau,
        LHEPart = nanoDQM.vplots.LHEPart,
        LHEScaleWeight = nanoDQM.vplots.LHEScaleWeight,
        LHEPdfWeight = nanoDQM.vplots.LHEPdfWeight,
        PSWeight = nanoDQM.vplots.PSWeight,
    )
)

from DQMServices.Core.DQMQualityTester import DQMQualityTester
nanoDQMQTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('PhysicsTools/NanoAOD/test/dqmQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    testInEventloop = cms.untracked.bool(False),
    qtestOnEndLumi = cms.untracked.bool(False),
    verboseQT =  cms.untracked.bool(True)
)

nanogenHarvest = cms.Sequence( nanoDQMQTester )
