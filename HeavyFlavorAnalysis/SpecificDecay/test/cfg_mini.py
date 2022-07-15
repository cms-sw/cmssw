import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

process = cms.Process("bphAnalysis")

patAlgosToolsTask = getPatAlgosToolsTask(process)

#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
patAlgosToolsTask.add(process.MEtoEDMConverter)
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring(
#
### use this to access the nearest copy of the input file, querying the catalog
    '/store/mc/RunIISpring15DR74/BsToJpsiPhi_BMuonFilter_TuneCUEP8M1_13TeV-pythia8-evtgen/MINIAODSIM/Asympt25nsRaw_MCRUN2_74_V9-v1/50000/0E685515-8661-E511-8274-00259073E3B6.root'
#
### use this to access the input file if by any reason you want to specify 
### the data server
#    'root://xrootd-cms.infn.it//store/mc/RunIISpring15DR74/BsToJpsiPhi_BMuonFilter_TuneCUEP8M1_13TeV-pythia8-evtgen/MINIAODSIM/Asympt25nsRaw_MCRUN2_74_V9-v1/50000/0E685515-8661-E511-8274-00259073E3B6.root'
#
### use this to access an input file locally available
#    'file:/...complete_file_path.../0E685515-8661-E511-8274-00259073E3B6.root'
))

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.testBPHSpecificDecay = cms.EDAnalyzer('TestBPHSpecificDecay',
    patMuonLabel = cms.string('slimmedMuons::PAT'),
    pcCandsLabel = cms.string('packedPFCandidates::PAT'),
    outDump = cms.string('dump_mini.txt'),
    outHist = cms.string('hist_mini.root')
)

process.p = cms.Path(
    process.testBPHSpecificDecay,
    patAlgosToolsTask
)

