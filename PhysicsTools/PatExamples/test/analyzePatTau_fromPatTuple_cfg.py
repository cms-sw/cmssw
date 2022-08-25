import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process('analyzePatTau')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100
#process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff')
process.GlobalTag.globaltag = cms.string( autoCond[ 'phase1_2022_realistic' ] )

process.maxEvents = cms.untracked.PSet(            
    input = cms.untracked.int32(-1)    
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:patTuple.root'
    )
)

process.maxEvents = cms.untracked.PSet(            
    input = cms.untracked.int32(-1)    
)

process.analyzePatTau = cms.EDAnalyzer("PatTauAnalyzer",
    src = cms.InputTag('cleanLayer1Taus'),
    requireGenTauMatch = cms.bool(True),
    discrByLeadTrack = cms.string("leadingTrackPtCut"),
    discrByIso = cms.string("byIsolation"),
    discrByTaNC = cms.string("byTaNCfrHalfPercent")
)

process.TFileService = cms.Service("TFileService", 
    fileName = cms.string('patTau_Histograms.root')
)

process.p = cms.Path( process.analyzePatTau )

# print-out all python configuration parameter information
#print process.dumpPython()
