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
process.GlobalTag.globaltag ='74X_dataRun2_v2'

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/0C806DA1-A127-E511-B9B2-02163E012601.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/0C8D29B7-9A27-E511-A7F2-02163E013770.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/0E183999-9E27-E511-BB8F-02163E012183.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/0E3D3941-9A27-E511-809E-02163E012183.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/0ED47BB7-9A27-E511-BA69-02163E012704.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/10176B03-9D27-E511-9D8B-02163E012595.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/104F51D7-9B27-E511-A02A-02163E01387D.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/1246E41C-9827-E511-8CD1-02163E011836.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/126A46B1-9627-E511-9691-02163E012BD2.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/12EBF600-9D27-E511-AF26-02163E0143AA.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/140C43DE-A027-E511-AA82-02163E0133B5.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/149CC210-9627-E511-BD0F-02163E011D23.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/163CE7FB-9827-E511-8257-02163E0133F9.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/16579E06-9E27-E511-BB3D-02163E01267F.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/16718FEC-A227-E511-8696-02163E012A7F.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/183354BC-9A27-E511-AFF8-02163E0133AD.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/18B6E10D-9827-E511-9F24-02163E013901.root',
       '/store/data/Run2015B/JetHT/RECO/PromptReco-v1/000/251/252/00000/18EFD43F-9E27-E511-8134-02163E013576.root'
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
                     #process.dump*
                     process.jetMETDQMOfflineSource*
                     process.dqmSaver
                     )
