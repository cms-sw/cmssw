import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

process = cms.Process("StudyCaloResponse", Run2_2018)

process.load("Calibration.IsolatedParticles.studyCaloResponse_cfi")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff')
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,'104X_dataRun2_v1', '')

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10000)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/data/Run2018B/MuonEGammaTOTEM/RECO/28Feb2019_resub-v1/260000/01B1233D-979E-F34F-A16F-308C41C36191.root',
    )
                            )

process.studyCaloResponse.verbosity = 0
process.studyCaloResponse.newNames = ["HLT_L1SingleMu_"]
#process.studyCaloResponse.newNames = ["HLT_L1DoubleJet_"]
process.studyCaloResponse.vetoMuon  = True
process.studyCaloResponse.vetoEcal  = True

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('studyCaloResponseMu.root')
                                   )

process.p = cms.Path(process.studyCaloResponse)
