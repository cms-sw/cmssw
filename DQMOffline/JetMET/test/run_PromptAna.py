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
process.GlobalTag.globaltag ='GR_R_74_V12A'

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       #'/store/relval/CMSSW_7_4_6_patch1/RelValProdTTbar_13/MINIAODSIM/MCRUN2_74_V9_unsch-v1/00000/28F53E5E-321D-E511-AEF1-0026189438F7.root',
       #'/store/relval/CMSSW_7_4_6_patch1/RelValProdTTbar_13/MINIAODSIM/MCRUN2_74_V9_unsch-v1/00000/4236E25F-321D-E511-92B6-0026189438B0.root' 
       #'/store/relval/CMSSW_7_4_3_patch1/JetHT/RECO/GR_R_74_V12A_unsch_RelVal_jet2012D-v1/00000/00648F9F-9D06-E511-A11C-0026189438C9.root',
       #'/store/relval/CMSSW_7_4_3_patch1/JetHT/RECO/GR_R_74_V12A_unsch_RelVal_jet2012D-v1/00000/026D63AD-A606-E511-B290-00261894386B.root',
       '/store/relval/CMSSW_7_4_6/RelValZMM_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V9-v2/00000/28CBB168-411A-E511-B2A2-002618943869.root'
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

process.p = cms.Path(                    #process.dump*
                     #process.jetMETDQMOfflineSourceMiniAOD*
                     #for cosmic data and MC
                     #process.jetMETDQMOfflineSourceCosmic*
                     #for Data and MC pp and HI
                     process.jetMETDQMOfflineSource*
                     process.dqmSaver
                     )
