import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("JetMETDQMOffline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

from Configuration.StandardSequences.GeometryRecoDB_cff import *
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")

#for data in 720pre7
process.GlobalTag.globaltag ='GR_R_74_V0A::All'

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       #for data
      #'/store/relval/CMSSW_7_4_0_pre6/JetHT/RECO/GR_R_74_V0A_RelVal_jet2012D-v1/00000/000FEFFF-CCA8-E411-BB29-003048FF9AC6.root',
       #for cosmics
       #'/store/data/Commissioning2014/Cosmics/RECO/PromptReco-v4/000/228/734/00000/10C180A7-2866-E411-B6F9-02163E010F8C.root',
       #for MC
       #'/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/22A79853-D85E-E411-BAA9-02163E00C055.root',
       #'/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/28923A16-C95E-E411-871C-02163E00FFCE.root',
       #'/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/307F76E6-E05E-E411-90AF-02163E00B036.root',
       #'/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/4E03E1A5-CE5E-E411-AE0F-02163E008BE3.root',
       #'/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/689DCC5B-D35E-E411-A720-02163E00D13A.root',
       #'/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/CC3F6060-DA5E-E411-BA7C-02163E0105B8.root',
       #'/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/D470466A-C55E-E411-A382-02163E00EB5D.root',
       #'/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/E2A34427-E75E-E411-ABBA-02163E008DD3.root',
       #'/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS172_V16-v1/00000/FCE96BE5-F15E-E411-BD38-02163E00D13A.root' 
       #for MINIAODtests 
       '/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/MINIAODSIM/PU50ns_PRE_LS172_V16-v1/00000/9886ACB4-F45E-E411-9E5D-02163E00F01E.root' 
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
                     process.jetMETDQMOfflineSourceMiniAOD*
                     #for cosmic data and MC
                     #process.jetMETDQMOfflineSourceCosmic*
                     #for Data and MC pp and HI
                     #process.jetMETDQMOfflineSource*
                     process.dqmSaver
                     )
