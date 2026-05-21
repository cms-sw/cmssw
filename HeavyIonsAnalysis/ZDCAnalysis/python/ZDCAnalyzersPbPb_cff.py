import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.ZDCAnalysis.zdcrecoRun3_cfi import *
from HeavyIonsAnalysis.ZDCAnalysis.ZDCRecHitAnalyzerHC_cfi import *

zdcrecoRun3.correctionMethodEM = cms.int32(0)
zdcrecoRun3.correctionMethodHAD = cms.int32(0)
zdcrecoRun3.ootpuRatioEM = cms.double(-1.0)
zdcrecoRun3.ootpuRatioHAD = cms.double(-1.0)
zdcrecoRun3.ootpuFracEM = cms.double(1.0)
zdcrecoRun3.ootpuFracHAD = cms.double(1.0)

zdcSequencePbPb = cms.Sequence(zdcrecoRun3 + zdcanalyzer)
