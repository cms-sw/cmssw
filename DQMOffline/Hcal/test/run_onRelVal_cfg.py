import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("hcalval")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']
#process.GlobalTag.globaltag = "GR_R_60_V1::All"

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_6_0_0_pre8-GR_R_60_V1_RelVal_mb2012A/MinimumBias/RECO/v1/0000/E0F52BA5-6CCC-E111-B31F-002618943971.root'
      ),
    inputCommands = cms.untracked.vstring('keep *', 'drop *_MEtoEDMConverter_*_*')
)

process.FEVT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
     fileName = cms.untracked.string("HcalValHarvestingEDM.root")
)

##########
process.load("DQMOffline.Hcal.HcalDQMOfflineSequence_cff")

process.calotowersAnalyzer.outputFile  = cms.untracked.string('CaloTowersValidationRelVal.root')
process.hcalNoiseRates.outputFile      = cms.untracked.string('NoiseRatesRelVal.root')
process.hcalRecHitsAnalyzer.outputFile = cms.untracked.string('HcalRecHitValidationRelVal.root')

##########
process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HcalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow

##########
process.load("DQMOffline.Hcal.HcalDQMOfflinePostProcessor_cff")

process.calotowersDQMClient.outputFile  = cms.untracked.string('CaloTowersHarvestingME.root')
process.hcalNoiseRatesClient.outputFile = cms.untracked.string('NoiseRatesHarvestingME.root')
process.hcalRecHitsDQMClient.outputFile = cms.untracked.string('HcalRecHitsHarvestingME.root')

##########
process.load("DQMServices.Components.DQMStoreStats_cfi")

##########
process.p2 = cms.Path( process.HcalDQMOfflineSequence
                       * process.HcalDQMOfflinePostProcessor
                       * process.dqmStoreStats
                       * process.dqmSaver)
