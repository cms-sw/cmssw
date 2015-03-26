import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("JetMETDQMOffline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

from Configuration.StandardSequences.GeometryRecoDB_cff import *
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")

#for data in 720pre7
process.GlobalTag.globaltag ='GR_R_74_V1::All'

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       #for data
        '/store/relval/CMSSW_7_3_2_patch1/JetHT/RECO/GR_R_73_V0_HcalExtValid_RelVal_jet2012D-v2/00000/0036492C-3BB5-E411-ADED-0025905A6136.root',
        '/store/relval/CMSSW_7_3_2_patch1/JetHT/RECO/GR_R_73_V0_HcalExtValid_RelVal_jet2012D-v2/00000/0081B579-2CB5-E411-AB79-0025905A60B8.root',
        '/store/relval/CMSSW_7_3_2_patch1/JetHT/RECO/GR_R_73_V0_HcalExtValid_RelVal_jet2012D-v2/00000/00CD85C8-34B5-E411-990F-0025905A612C.root',
        '/store/relval/CMSSW_7_3_2_patch1/JetHT/RECO/GR_R_73_V0_HcalExtValid_RelVal_jet2012D-v2/00000/00F53EF1-25B5-E411-B905-0025905A6094.root',
        '/store/relval/CMSSW_7_3_2_patch1/JetHT/RECO/GR_R_73_V0_HcalExtValid_RelVal_jet2012D-v2/00000/023CAC47-2DB5-E411-BD56-0025905A612E.root',
        '/store/relval/CMSSW_7_3_2_patch1/JetHT/RECO/GR_R_73_V0_HcalExtValid_RelVal_jet2012D-v2/00000/02A0F516-38B5-E411-8233-0025905A610C.root',
        '/store/relval/CMSSW_7_3_2_patch1/JetHT/RECO/GR_R_73_V0_HcalExtValid_RelVal_jet2012D-v2/00000/02C3E687-26B5-E411-9B42-0025905964BA.root',
        '/store/relval/CMSSW_7_3_2_patch1/JetHT/RECO/GR_R_73_V0_HcalExtValid_RelVal_jet2012D-v2/00000/067B87D5-2BB5-E411-BE4F-002590593872.root'
        '/store/relval/CMSSW_7_4_0_pre6/JetHT/RECO/GR_R_74_V0A_RelVal_jet2012D-v1/00000/000FEFFF-CCA8-E411-BB29-003048FF9AC6.root',
       #for cosmics
       #'/store/data/Commissioning2014/Cosmics/RECO/PromptReco-v4/000/228/734/00000/10C180A7-2866-E411-B6F9-02163E010F8C.root',
       #for MC
       #'/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/MCRUN2_74_V1-v1/00000/2E97B200-D0A8-E411-BE99-0025905A60B8.root',
       #'/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/MCRUN2_74_V1-v1/00000/76F6A5D4-DEA8-E411-BBE9-0026189437F0.root',
       #'/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/MCRUN2_74_V1-v1/00000/82D11D79-D1A8-E411-BF67-003048FFCBFC.root',
       #'/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/MCRUN2_74_V1-v1/00000/B0DB64D8-DEA8-E411-B307-0025905964A2.root'
       #for MINIAODtests 
       #'/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/MINIAODSIM/MCRUN2_74_V1-v1/00000/58D65E53-E5A8-E411-BC5E-002354EF3BDF.root',
       #'/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/MINIAODSIM/MCRUN2_74_V1-v1/00000/D0654655-E5A8-E411-97EA-0025905964C2.root'
       #for HI tests       
       #'/store/relval/CMSSW_7_3_0_pre1/RelValQCD_Pt_80_120_13_HI/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/5C15CC80-0B5A-E411-AF4B-02163E00ECD2.root',
       #'/store/relval/CMSSW_7_3_0_pre1/RelValQCD_Pt_80_120_13_HI/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/FC51FED6-B559-E411-9131-02163E006D72.root'
       #cosmics 
       #'/store/relval/CMSSW_7_3_0_pre1/Cosmics/RECO/PRE_R_72_V10A_RelVal_cos2011A-v1/00000/06393A70-DB59-E411-865C-0025905A612C.root',
       #'/store/relval/CMSSW_7_3_0_pre1/Cosmics/RECO/PRE_R_72_V10A_RelVal_cos2011A-v1/00000/2838CB6D-DB59-E411-8001-0025905A611E.root
       ] );


secFiles.extend( [
               ] )

#
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 1000)
)

process.load("DQMOffline.JetMET.dataCertificationJetMET_cff")

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

process.p = cms.Path(                    #process.dump*
                     #process.jetMETDQMOfflineSourceMiniAOD*
                     #for cosmic data and MC
                     #process.jetMETDQMOfflineSourceCosmic*
                     #for Data and MC pp and HI
                     process.jetMETDQMOfflineSource*
                     process.dataCertificationJetMETSequence*
                     process.dqmSaver
                     )
