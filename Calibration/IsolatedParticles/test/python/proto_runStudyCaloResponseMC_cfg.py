import FWCore.ParameterSet.Config as cms

process = cms.Process("StudyCaloResponse")

process.load("Calibration.IsolatedParticles.studyCaloResponse_cfi")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff')
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

if hasattr(process,'MessageLogger'):
    process.MessageLogger.IsoTrack=dict()

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10000)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
'/store/mc/RunIILowPUAutumn18MiniAOD/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/MINIAODSIM/NoPU_102X_upgrade2018_realistic_v15-v2/00000/0227F562-C82C-814E-9D51-F8895E245DD5.root',)
                            )

process.studyCaloResponse.verbosity = 110
process.studyCaloResponse.vetoMuon = True
process.studyCaloResponse.vetoEcal = True
process.studyCaloResponse.triggers = []

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('studyCaloResponseMC.root')
                                   )

process.p = cms.Path(process.studyCaloResponse)
