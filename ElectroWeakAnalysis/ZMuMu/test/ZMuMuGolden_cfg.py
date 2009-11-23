import FWCore.ParameterSet.Config as cms

process = cms.Process("EwkZMuMuGolden")

process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuGolden_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V3::All')
process.load("Configuration.StandardSequences.MagneticField_cff")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)



process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
'file:/tmp/degrutto/0ABB0814-C082-DE11-9AB7-003048D4767C.root',
   # 'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_4_0_pre1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V8-v1/0007/CAE2081C-48B5-DE11-9161-001D09F29321.root',
    )
)


process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("ewkZMuMuGolden.root")
)

process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)

process.ewkZMuMuGoldenPath = cms.Path(
    process.ewkZMuMuGoldenSequence 
)



process.endPath = cms.EndPath( 
    process.eventInfo 
)
