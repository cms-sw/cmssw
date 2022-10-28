import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process('analyzePatTau')

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100
#process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = cms.string( autoCond[ 'phase1_2022_realistic' ] )
process.load('PhysicsTools.PatAlgos.patSequences_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_4_2_8'
                        , relVal        = 'RelValTTbar'
                        , globalTag     = 'START42_V12'
                        , numberOfFiles = 1
                        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# switch to HPS + TaNC combined tau id. algorithm
from PhysicsTools.PatAlgos.tools.tauTools import *
switchToPFTauHPSpTaNC(process)

process.analyzePatTau = cms.EDAnalyzer("PatTauAnalyzer",
    src = cms.InputTag('cleanPatTaus'),
    requireGenTauMatch = cms.bool(True),
    discrByLeadTrack = cms.string("leadingTrackPtCut"),
    discrByIso = cms.string("byHPSloose"),
    discrByTaNC = cms.string("byTaNCmedium")
)

# disable preselection on pat::Tau objects
# (neccessary in order to make efficiency plots)
process.cleanPatTaus.preselection = cms.string('')

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('patTau_Histograms.root')
)

process.p = cms.Path( process.patDefaultSequence + process.analyzePatTau )

# print-out all python configuration parameter information
#print process.dumpPython()
