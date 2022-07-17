#This file is meant to test legacy modules that may still be used by some analyses
from FWCore.ParameterSet.Config import *

process = Process("CandSelectorTest")

process.maxEvents = untracked.PSet( input = untracked.int32(10) )

process.source = Source("PoolSource",
  fileNames = untracked.vstring("/store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/00000/9F030B27-DBB0-EE46-A8FB-64FE0C417EE9.root")
)

process.genmuons = EDFilter("CandViewShallowCloneProducer",
  src = InputTag("packedPFCandidates"),
  cut = string("abs(pdgId)==13")
)

process.Ztomumu = EDProducer("CandViewCombiner",
    decay = string("genmuons@+ genmuons@-"),
    cut = string("0.0 < mass < 200.0")
)

process.out = OutputModule("PoolOutputModule",
    fileName = untracked.string("test.root"),
    outputCommands = untracked.vstring(
        'drop *',
        'keep *_Ztomumu_*_*'
    )
)

task = Task(process.genmuons,process.Ztomumu)
process.path = Path(task)
process.endpath = EndPath(process.out)

