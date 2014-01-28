import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process('analyzePatTau')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100
#process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = cms.string('STARTUP31X_V1::All')

process.load('PhysicsTools.PatAlgos.patSequences_cff')

process.maxEvents = cms.untracked.PSet(            
    input = cms.untracked.int32(-1)    
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_3_6/RelValZTT/GEN-SIM-RECO/STARTUP3X_V8H-v1/0009/24052BB1-9EE4-DE11-87C2-002618943957.root'
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

# disable preselection on pat::Tau objects
# (neccessary in order to make efficiency plots)
process.cleanLayer1Taus.preselection = cms.string('')

process.TFileService = cms.Service("TFileService", 
    fileName = cms.string('patTau_Histograms.root')
)

process.p = cms.Path( process.patDefaultSequence + process.analyzePatTau )

# print-out all python configuration parameter information
#print process.dumpPython()
