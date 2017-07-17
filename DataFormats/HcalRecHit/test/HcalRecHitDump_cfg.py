# Run with something like the following:
# cmsRun HcalRecHitDump_cfg.py |& grep ^++HBHE++ | sed 's/to -120 ns/to 0 ns/g' > dump.log

import os
import FWCore.ParameterSet.Config as cms

# Get the input file from the environment
inputfile = os.environ['INPUTFILE']

process = cms.Process('Test')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:' + inputfile)
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
                                38,37,36,35,34,33,32,-1,46,45,44,43,42,41,40,-1,54,53,52,51,50,49,48,-1,62,61,60,59,58,57,56,-1,\
                                70,69,68,67,66,65,64,-1,78,77,76,75,74,73,72,-1,86,85,84,83,82,81,80,-1,94,93,92,91,90,89,88,-1,\
                                121,120)
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
                                70,69,68,67,66,65,64,-1,77,76,75,74,73,72,71,-1,84,83,82,81,80,79,78,-1,91,90,89,88,87,86,85,-1,\
                                38,37,36,35,34,33,32,-1,45,44,43,42,41,40,39,-1,52,51,50,49,48,47,46,-1,59,58,57,56,55,54,53,-1,\
                                61,60)
)

process.p = cms.Path(process.dumpPhase1)
