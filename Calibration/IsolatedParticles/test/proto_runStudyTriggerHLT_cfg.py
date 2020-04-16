from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.coreTools import *

process = cms.Process("StudyTriggerHLT")

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("Calibration.IsolatedParticles.studyTriggerHLT_cfi")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run1_data']

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10000)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
#    '/store/data/Run2012A/MinimumBias/RECO/13Jul2012-v1/00000/001767E2-FFCF-E111-BF8A-003048FFD76E.root'
    '/store/data/Run2012A/MinimumBias/RECO/13Jul2012-v1/00001/CAFCD70A-6BD0-E111-B8AD-003048678B1A.root',
    )
                            )

process.studyTriggerHLT.verbosity = 0

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('studyTriggerHLT.root')
                                   )

process.p = cms.Path(process.studyTriggerHLT)
