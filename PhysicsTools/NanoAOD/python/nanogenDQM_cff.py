import FWCore.ParameterSet.Config as cms
import copy

from PhysicsTools.NanoAOD.nanoDQM_cfi import nanoDQM
from PhysicsTools.NanoAOD.nanoDQM_tools_cff import *


nanogenDQM = nanoDQM.clone()

nanogenDQM.vplots = [nanoDQM.vplots.GenDressedLepton,
        nanoDQM.vplots.GenIsolatedPhoton,
        nanoDQM.vplots.GenJet,
        nanoDQM.vplots.GenJetAK8,
        nanoDQM.vplots.GenMET,
        nanoDQM.vplots.GenPart, 
        nanoDQM.vplots.GenVisTau,
    ]

from DQMServices.Core.DQMQualityTester import DQMQualityTester
nanoDQMQTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('PhysicsTools/NanoAOD/test/dqmQualityTests.xml'),
    prescaleFactor = cms.untracked.int32(1),                               
    testInEventloop = cms.untracked.bool(False),
    qtestOnEndLumi = cms.untracked.bool(False),
    verboseQT =  cms.untracked.bool(True)
)

nanogenHarvest = cms.Sequence( nanoDQMQTester )
