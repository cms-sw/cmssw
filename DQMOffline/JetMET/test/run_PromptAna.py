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
process.GlobalTag.globaltag ='75X_dataRun1_v2'

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_7_5_0_pre6/JetHT/RECO/75X_dataRun1_v2_RelVal_jet2012D-v1/00000/003E50FE-F71B-E511-B849-002590593902.root',
       '/store/relval/CMSSW_7_5_0_pre6/JetHT/RECO/75X_dataRun1_v2_RelVal_jet2012D-v1/00000/0212B89F-401C-E511-90B7-0025905964A2.root',
       '/store/relval/CMSSW_7_5_0_pre6/JetHT/RECO/75X_dataRun1_v2_RelVal_jet2012D-v1/00000/02925C10-FC1B-E511-A4B4-0025905B85A2.root',
       '/store/relval/CMSSW_7_5_0_pre6/JetHT/RECO/75X_dataRun1_v2_RelVal_jet2012D-v1/00000/040DBAE5-FB1B-E511-9854-0025905B85D0.root',
       '/store/relval/CMSSW_7_5_0_pre6/JetHT/RECO/75X_dataRun1_v2_RelVal_jet2012D-v1/00000/0658D146-2C1C-E511-BCA1-0025905A48F2.root',
       '/store/relval/CMSSW_7_5_0_pre6/JetHT/RECO/75X_dataRun1_v2_RelVal_jet2012D-v1/00000/0A06BB23-401C-E511-ACE9-0025905A610C.root',
       '/store/relval/CMSSW_7_5_0_pre6/JetHT/RECO/75X_dataRun1_v2_RelVal_jet2012D-v1/00000/0AA1F5F7-FA1B-E511-B761-0025905A48FC.root',
       '/store/relval/CMSSW_7_5_0_pre6/JetHT/RECO/75X_dataRun1_v2_RelVal_jet2012D-v1/00000/0C963256-311C-E511-88DA-002618943842.root'
       ] );


secFiles.extend( [
               ] )

#
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
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
