import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.coreTools import *

process = cms.Process("StudyHLT")

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("Calibration.IsolatedParticles.studyHLT_cfi")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run1_data']

################# CommandLine Parsing
import os
import sys
import FWCore.ParameterSet.VarParsing as VarParsing
# setup 'standard'  options
options = VarParsing.VarParsing ('standard')
options.register ( "TrigNames",
                   "HLT",
                   VarParsing.VarParsing.multiplicity.list, # singleton or list
                   VarParsing.VarParsing.varType.string,          # string, int, or float
                   "HLT names")
options.parseArguments()
print options.TrigNames

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10000)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
#    '/store/data/Run2012A/MinimumBias/RECO/13Jul2012-v1/00000/001767E2-FFCF-E111-BF8A-003048FFD76E.root'
    '/store/data/Run2012A/MinimumBias/RECO/13Jul2012-v1/00001/CAFCD70A-6BD0-E111-B8AD-003048678B1A.root',
    '/store/data/Run2012A/MinimumBias/RECO/13Jul2012-v1/00001/BAECADE6-79D0-E111-B4A4-00261894382A.root',
    '/store/data/Run2012A/MinimumBias/RECO/13Jul2012-v1/00001/BA276120-6BD0-E111-A2D8-00304867C16A.root',
    '/store/data/Run2012A/MinimumBias/RECO/13Jul2012-v1/00000/1E2DE509-62D0-E111-8A5C-0026189437F9.root'
    )
                            )

process.StudyHLT.Verbosity = 0
process.StudyHLT.Triggers  = options.TrigNames

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('StudyHLT.root')
                                   )

process.p = cms.Path(process.StudyHLT)
