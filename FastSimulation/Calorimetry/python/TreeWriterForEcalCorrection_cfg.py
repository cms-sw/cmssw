import FWCore.ParameterSet.Config as cms
process = cms.Process("ANA")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10000
process.MessageLogger.cerr.default.limit = 100000000


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

from python.fastFileList import filelist
#from python.fullFileList import filelist

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'/store/user/kiesel/SingleElectronGun_E80_fast/SingleElectronGun_E80_fast/a88c9ccbd07595a7d33ae9d4d6a917a0/out_2_1_X2L.root',
        #'/store/user/kiesel/SingleElectronGun_E80_full/SingleElectronGun_E80_full/1d73d2566f2f721e3a7146e7729d743b/out_48_1_kkW.root'
        filelist
    )
)

outFileName = "out_tree.root"

# guess the output name from the 1. input name
fn0 = process.source.fileNames[0]
if "full" in fn0 or "Full" in fn0:
    outFileName = "full_tree.root"
if "fast" in fn0 or "Fast" in fn0:
    outFileName = "fast_tree.root"

process.TFileService = cms.Service("TFileService",
    fileName = cms.string( outFileName )
)

process.load("Configuration.StandardSequences.Analysis_cff")

process.treeWriterForEcalCorrection = cms.EDAnalyzer('TreeWriterForEcalCorrection')
process.p = cms.Path( process.treeWriterForEcalCorrection )

