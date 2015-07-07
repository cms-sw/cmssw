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
process.GlobalTag.globaltag ='75X_dataRun2_v2'

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_7_5_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PUpmx25ns_MCRUN2_75_V5-v1/00000/1062212F-900C-E511-A3F1-0025905A6060.root',
       '/store/relval/CMSSW_7_5_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PUpmx25ns_MCRUN2_75_V5-v1/00000/32BCC3D8-120D-E511-B559-002618943961.root',
       '/store/relval/CMSSW_7_5_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PUpmx25ns_MCRUN2_75_V5-v1/00000/3E502ECF-8F0C-E511-8FD6-0025905B85AE.root',
       '/store/relval/CMSSW_7_5_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PUpmx25ns_MCRUN2_75_V5-v1/00000/543CCBA0-8D0C-E511-B87F-002590593920.root',
       '/store/relval/CMSSW_7_5_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PUpmx25ns_MCRUN2_75_V5-v1/00000/80AF9DD7-8F0C-E511-86B0-0025905A6070.root',
       '/store/relval/CMSSW_7_5_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PUpmx25ns_MCRUN2_75_V5-v1/00000/88F3F0E7-8F0C-E511-ADF4-002618943810.root',
       '/store/relval/CMSSW_7_5_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PUpmx25ns_MCRUN2_75_V5-v1/00000/B4B94F85-AB0C-E511-B5A3-0025905B85D0.root',
       '/store/relval/CMSSW_7_5_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PUpmx25ns_MCRUN2_75_V5-v1/00000/C411BEDE-8F0C-E511-8E91-0025905B85D8.root'
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
                     #process.dump*
                     #process.dataCertificationJetMETSequence*
                     process.dqmSaver
                     )
