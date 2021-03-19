import FWCore.ParameterSet.Config as cms
from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
from Configuration.AlCa.GlobalTag import GlobalTag

process = cms.Process("TestGsfElectronConversionFinder")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1000))

process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(
        "/store/mc/RunIISummer17DRPremix/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/AODSIM/92X_upgrade2017_realistic_v10_ext1-v1/110000/001907F5-C185-E711-B1DD-02163E014A5B.root"
    ),
)

process.testGsfElectronConversionFinder = cms.EDAnalyzer("TestGsfElectronConversionFinder")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    files = cms.untracked.PSet(
        conversions = cms.untracked.PSet(
            threshold = cms.untracked.string('INFO')
        )
    )
)

process.p = cms.Path(process.testGsfElectronConversionFinder)
