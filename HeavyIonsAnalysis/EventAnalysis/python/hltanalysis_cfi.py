import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.EventAnalysis.dummybranches_cff import *

hltanalysis = cms.EDAnalyzer(
    'TriggerAnalyzer',
    HLTProcessName = cms.string('HLT'),
    hltresults = cms.InputTag('TriggerResults::HLT'),
    l1results = cms.InputTag('gtStage2Digis'),
    hltdummybranches = dummy_branches_for_PbPb_2023_HLT,
    l1dummybranches = dummy_branches_for_PbPb_2023_L1,
    hltPSProvCfg=cms.PSet(stageL1Trigger = cms.uint32(2)),
    )
