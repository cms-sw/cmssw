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
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/00088BD6-813C-E511-B574-0025905964BC.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/0253073D-7F3C-E511-BC5B-0026189438F7.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/04CACE3D-7F3C-E511-B0E8-002618943925.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/04E0F04B-7F3C-E511-A19D-002590593876.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/060DFA85-803C-E511-9E12-0025905A60F4.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/0862E1A9-823C-E511-AF36-0025905A6088.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/0AD74E86-803C-E511-A715-0025905B85AA.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/0C2164D3-813C-E511-8594-0025905A48F0.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/0C487CE9-813C-E511-9D45-0025905A610A.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/0E0CAFDB-7D3C-E511-B7A1-0025905B855C.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/0E657740-7F3C-E511-8F5E-003048FFD770.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/10288AE2-7D3C-E511-922D-0025905B85D6.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/10474781-853C-E511-9685-0025905A60F8.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/105B72DC-7C3C-E511-9C63-0026189438F3.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/12322589-803C-E511-9010-0025905A60F4.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/127639DA-813C-E511-BF5D-002618943874.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/144D2FEC-7D3C-E511-92C6-003048FFD720.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/160D4B46-7D3C-E511-89E8-0025905B860E.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/16467F99-803C-E511-9EA4-003048FFCC18.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/164A1983-803C-E511-B28F-0025905A6110.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/16717A4C-7F3C-E511-BCFB-003048FFD760.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/18641BAD-7E3C-E511-970C-0025905A609A.root',
       #'/store/relval/CMSSW_7_4_8_patch1/JetHT/RECO/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/18798CEC-813C-E511-B01F-0025905A6138.root'
       #'/store/relval/CMSSW_7_4_6_patch1/RelValProdTTbar_13/MINIAODSIM/MCRUN2_74_V9_unsch-v1/00000/28F53E5E-321D-E511-AEF1-0026189438F7.root',
       #'/store/relval/CMSSW_7_4_6_patch1/RelValProdTTbar_13/MINIAODSIM/MCRUN2_74_V9_unsch-v1/00000/4236E25F-321D-E511-92B6-0026189438B0.root' 
       #'/store/relval/CMSSW_7_4_8_patch1/RelValTTbar_13/MINIAODSIM/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/3E8E7CFC-503C-E511-B57E-0025905A60A6.root',
       #'/store/relval/CMSSW_7_4_8_patch1/RelValTTbar_13/MINIAODSIM/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/DA8BC5FB-503C-E511-AD75-0025905A48D8.root' 
       #'/store/relval/CMSSW_7_4_3_patch1/JetHT/RECO/GR_R_74_V12A_unsch_RelVal_jet2012D-v1/00000/00648F9F-9D06-E511-A11C-0026189438C9.root',
       #'/store/relval/CMSSW_7_4_3_patch1/JetHT/RECO/GR_R_74_V12A_unsch_RelVal_jet2012D-v1/00000/026D63AD-A606-E511-B290-00261894386B.root',
       '/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/5CEA33B7-873C-E511-BE51-0025905A60DA.root',
       '/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/8226D0B8-873C-E511-ACD3-0025905B85AE.root',
       '/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/924546B9-873C-E511-9B90-0025905B8576.root',
       '/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/A2937EB9-873C-E511-BA25-0025905B8598.root',
       '/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/AA4690B5-873C-E511-B398-0025905A60A8.root',
       '/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/BA0637B7-873C-E511-AE94-0025905A60B2.root',
       '/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/C6DC17B8-873C-E511-B34C-0025905A6084.root',
       '/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/CA8E9CB9-873C-E511-A795-0025905B8598.root',
       '/store/relval/CMSSW_7_4_8_patch1/JetHT/MINIAOD/GR_H_V57A_mulTrh_RelVal_jet2012D-v1/00000/F44521B8-873C-E511-8C81-0025905B85F6.root' 
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
                     process.jetMETDQMOfflineSourceMiniAOD*
                     #for cosmic data and MC
                     #process.jetMETDQMOfflineSourceCosmic*
                     #for Data and MC pp and HI
                     #process.jetMETDQMOfflineSource*
#                     process.dump*
                     process.dqmSaver
                     )
