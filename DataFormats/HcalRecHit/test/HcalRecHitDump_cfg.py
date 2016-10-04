# Run with something like the following:
# cmsRun HcalRecHitDump_cfg.py |& grep ^++HBHE++ | sed 's/to -120 ns/to 0 ns/g' > dump.log

import FWCore.ParameterSet.Config as cms

process = cms.Process('Test')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/i/igv/CMSSW_8_1_X_2016-09-26-2300/src/25.0_TTbar+TTbar+DIGI+RECOAlCaCalo+HARVEST+ALCATT/step3.root')
)

# Settings for rechits reconstructed by HBHEPhase1Reconstructor
process.dumpPhase1 = cms.EDAnalyzer(
    'HcalRecHitDump',
    hbhePrefix = cms.untracked.string("++HBHE++ "),
    # hfPrefix = cms.untracked.string("++HF++ "),
    # hfprePrefix = cms.untracked.string("++HFPre++ "),
    tagHBHE = cms.InputTag("hbheprereco"),
    tagPreHF = cms.InputTag("hfprereco"),
    tagHF = cms.InputTag("hfreco"),
    bits = cms.untracked.vint32(0,11,12,13,15,27,29,30,-1,\
                                32,33,34,35,36,37,38,40,41,42,43,44,45,46,48,49,50,51,52,53,54,56,57,58,59,60,61,62,-1,\
                                64,65,66,67,68,69,70,72,73,74,75,76,77,78,80,81,82,83,84,85,86,88,89,90,91,92,93,94,-1,\
                                120,121)
)

# Settings for rechits reconstructed by HcalHitReconstructor
process.dumpLegacy = cms.EDAnalyzer(
    'HcalRecHitDump',
    hbhePrefix = cms.untracked.string("++HBHE++ "),
    # hfPrefix = cms.untracked.string("++HF++ "),
    # hfprePrefix = cms.untracked.string("++HFPre++ "),
    tagHBHE = cms.InputTag("hbheprereco"),
    tagPreHF = cms.InputTag("hfprereco"),
    tagHF = cms.InputTag("hfreco"),
    bits = cms.untracked.vint32(0,11,12,13,15,27,29,30,-1,\
                                64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,-1,\
                                32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,-1,\
                                60,61)
)

# Change the path to "process.dumpLegacy" for dumping rechits produced by
# HcalHitReconstructor
process.p = cms.Path(process.dumpPhase1)
