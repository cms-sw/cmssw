import FWCore.ParameterSet.Config as cms

process = cms.Process("RaddamMuon")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")  
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoJets.Configuration.CaloTowersES_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag='101X_dataRun2_Prompt_v11'

process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load("Calibration.HcalCalibAlgos.hcalHBHEMuon_cfi")

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HBHEMuon=dict()

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'root://cms-xrd-global.cern.ch//store/data/Run2018C/SingleMuon/ALCARECO/HcalCalHBHEMuonFilter-PromptReco-v1/000/319/337/00000/004EC357-9184-E811-A588-FA163EFF1C10.root',
        )
                            )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("Validation.root")
)

process.hcalHBHEMuon.useRaw = 0
process.hcalHBHEMuon.unCorrect = True
process.hcalHBHEMuon.getCharge = True
process.hcalHBHEMuon.ignoreHECorr = False
process.hcalHBHEMuon.maxDepth = 7
process.hcalHBHEMuon.verbosity = 0
process.hcalTopologyIdeal.MergePosition = False

process.p = cms.Path(process.hcalHBHEMuon)
