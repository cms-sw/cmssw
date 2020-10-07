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
        LHEPart = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Count1D('_size', 20, 0, 20, 'LHE particles'),
                Plot1D('eta', 'eta', 20, -30000, 30000, 'eta'),
                Plot1D('pdgId', 'pdgId', 20, -6000, 6000, 'PDG id'),
                Plot1D('phi', 'phi', 20, -3.14159, 3.14159, 'phi'),
                Plot1D('pt', 'pt', 20, 0, 200, 'pt'),
            )
        ),
        LHEScaleWeight = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Count1D('_size', 20, 0, 20, 'LHE scale weights'),
                Plot1D('', '', 100, 0, 2, 'all weights'),
            )
        ),
        LHEPdfWeight = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Count1D('_size', 1000, 0, 2000, 'LHE PDF weights'),
                Plot1D('', '', 100, 0, 2, 'all weights'),
            )
        ),
        PSWeight = cms.PSet(
            sels = cms.PSet(),
            plots = cms.VPSet(
                Count1D('_size', 50, 0, 50, 'LHE PDF weights'),
                Plot1D('', '', 100, 0, 2, 'all weights'),
            )
        ),
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
