import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("JetMETDQMOffline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

from Configuration.StandardSequences.GeometryRecoDB_cff import *
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

#for data in 720pre7
process.GlobalTag.globaltag ='74X_dataRun2_Prompt_v3'

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/251/00000/00597BBF-8027-E511-8D9F-02163E014376.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/251/00000/008D4935-8127-E511-BD12-02163E0127DF.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/251/00000/00B58C93-8527-E511-9127-02163E014553.root'
       #'/store/backfill/1/data/Tier0_Test_SUPERBUNNIES_vocms047/JetHT/MINIAOD/PromptReco-v91/000/251/251/00000/6C03D331-3858-E511-9B7C-02163E017135.root',
       #'/store/backfill/1/data/Tier0_Test_SUPERBUNNIES_vocms047/JetHT/MINIAOD/PromptReco-v91/000/251/251/00000/9C949F3F-3858-E511-B785-02163E013750.root',
       #'/store/backfill/1/data/Tier0_Test_SUPERBUNNIES_vocms047/JetHT/MINIAOD/PromptReco-v91/000/251/251/00000/AAEAE841-3858-E511-A66E-02163E011989.root',
       #'/store/backfill/1/data/Tier0_Test_SUPERBUNNIES_vocms047/JetHT/MINIAOD/PromptReco-v91/000/251/251/00000/DEFB433D-3858-E511-A1C1-02163E0119B3.root' 
       #'/store/data/Run2015C/JetHT/MINIAOD/PromptReco-v1/000/254/790/00000/06947E9F-204A-E511-B627-02163E0137BA.root',
       #'/store/data/Run2015C/JetHT/MINIAOD/PromptReco-v1/000/254/790/00000/18D18896-204A-E511-B82D-02163E01190D.root',
       #'/store/data/Run2015C/JetHT/MINIAOD/PromptReco-v1/000/254/790/00000/260A1195-204A-E511-8627-02163E014125.root',
       #'/store/data/Run2015C/JetHT/MINIAOD/PromptReco-v1/000/254/790/00000/6A03E597-204A-E511-B2EA-02163E01418B.root',
       #'/store/data/Run2015C/JetHT/MINIAOD/PromptReco-v1/000/254/790/00000/7086989C-204A-E511-B943-02163E013409.root',
       #'/store/data/Run2015C/JetHT/MINIAOD/PromptReco-v1/000/254/790/00000/7CA25D9B-204A-E511-BF7A-02163E011955.root',
       #'/store/data/Run2015C/JetHT/MINIAOD/PromptReco-v1/000/254/790/00000/823FA0A1-204A-E511-887F-02163E01453E.root',
       #'/store/data/Run2015C/JetHT/MINIAOD/PromptReco-v1/000/254/790/00000/AA29DAA0-204A-E511-83FD-02163E0136D5.root',
       #'/store/data/Run2015C/JetHT/MINIAOD/PromptReco-v1/000/254/790/00000/BA4041A9-204A-E511-937D-02163E011E1D.root',
       #'/store/data/Run2015C/JetHT/MINIAOD/PromptReco-v1/000/254/790/00000/BE8974AB-204A-E511-9AB8-02163E01266D.root',
       #'/store/data/Run2015C/JetHT/MINIAOD/PromptReco-v1/000/254/790/00000/C42C66A7-204A-E511-A53D-02163E01448C.root',
       #'/store/data/Run2015C/JetHT/MINIAOD/PromptReco-v1/000/254/790/00000/F2098695-204A-E511-85CE-02163E014657.root',
       #'/store/data/Run2015C/JetHT/MINIAOD/PromptReco-v1/000/254/790/00000/F60BE6A7-204A-E511-AF26-02163E0146B3.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/5CEA33B7-873C-E511-BE51-0025905A60DA.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/8226D0B8-873C-E511-ACD3-0025905B85AE.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/924546B9-873C-E511-9B90-0025905B8576.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/A2937EB9-873C-E511-BA25-0025905B8598.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/AA4690B5-873C-E511-B398-0025905A60A8.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/BA0637B7-873C-E511-AE94-0025905A60B2.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/C6DC17B8-873C-E511-B34C-0025905A6084.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/CA8E9CB9-873C-E511-A795-0025905B8598.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/F44521B8-873C-E511-8C81-0025905B85F6.root' 
       ] );



secFiles.extend( [
               ] )

#
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load('Configuration/StandardSequences/EDMtoMEAtJobEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')
#CMSSW expects same sequence name for different modes, comment out unneeded modes
#pp MC
#process.load("DQMOffline.JetMET.jetMETDQMOfflineSourceMC_cff")
#pp data
process.load("DQMOffline.JetMET.jetMETDQMOfflineSource_cff")
#cosmic data
#process.load("DQMOffline.JetMET.jetMETDQMOfflineSourceCosmic_cff")
#cosmic MC
#process.load("DQMOffline.JetMET.jetMETDQMOfflineSourceCosmicMC_cff")
#for HI data
#process.load("DQMOffline.JetMET.jetMETDQMOfflineSourceHI_cff")
#for HI MC
#process.load("DQMOffline.JetMET.jetMETDQMOfflineSourceHIMC_cff")

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/JetMET/'+str(cmssw_version)+'/Harvesting'
process.dqmSaver.workflow = Workflow

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(
                     #process.jetMETDQMOfflineSourceMiniAOD*
                     #for cosmic data and MC
                     #process.jetMETDQMOfflineSourceCosmic*
                     #for Data and MC pp and HI
                     process.jetMETDQMOfflineSource*
                     #process.dump*
                     process.dqmSaver
                     )
