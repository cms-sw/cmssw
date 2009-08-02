import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source (
    "PoolSource",    
    fileNames = cms.untracked.vstring(
      'rfio:/castor/cern.ch/user/c/cbern/CMSSW312/Fast/aod_QCDForPF_Fast_0.root',
      'rfio:/castor/cern.ch/user/c/cbern/CMSSW312/Fast/aod_QCDForPF_Fast_1.root',
      'rfio:/castor/cern.ch/user/c/cbern/CMSSW312/Fast/aod_QCDForPF_Fast_10.root',
      'rfio:/castor/cern.ch/user/c/cbern/CMSSW312/Fast/aod_QCDForPF_Fast_2.root',
      'rfio:/castor/cern.ch/user/c/cbern/CMSSW312/Fast/aod_QCDForPF_Fast_3.root',
      'rfio:/castor/cern.ch/user/c/cbern/CMSSW312/Fast/aod_QCDForPF_Fast_4.root',
      'rfio:/castor/cern.ch/user/c/cbern/CMSSW312/Fast/aod_QCDForPF_Fast_5.root',
      'rfio:/castor/cern.ch/user/c/cbern/CMSSW312/Fast/aod_QCDForPF_Fast_6.root',
      'rfio:/castor/cern.ch/user/c/cbern/CMSSW312/Fast/aod_QCDForPF_Fast_7.root',
      'rfio:/castor/cern.ch/user/c/cbern/CMSSW312/Fast/aod_QCDForPF_Fast_8.root',
      'rfio:/castor/cern.ch/user/c/cbern/CMSSW312/Fast/aod_QCDForPF_Fast_9.root',
      ),
    secondaryFileNames = cms.untracked.vstring(),
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
    )


process.pfCandidateAnalyzer = cms.EDAnalyzer("PFCandidateAnalyzer",
    PFCandidates = cms.InputTag("particleFlow"),
    verbose = cms.untracked.bool(True),
    printBlocks = cms.untracked.bool(False)
)

process.load("FastSimulation.Configuration.EventContent_cff")
process.aod = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('aod.root')
)

process.outpath = cms.EndPath(process.aod )


process.p = cms.Path(process.pfCandidateAnalyzer)


