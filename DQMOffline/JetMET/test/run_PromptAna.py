import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("JetMETDQMOffline")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

from Configuration.StandardSequences.GeometryRecoDB_cff import *
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.Services_cff')

#for data in 720pre7
process.GlobalTag.globaltag ='92X_dataRun2_Prompt_v4'

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
        #'file:/eos/cms/store/data/Run2017E/ZeroBias/RECO/09Oct2017-v1/00000/20494CF3-10AD-E711-ACB6-FA163E42B482.root',
        #'file:/eos/cms/store/data/Run2017E/ZeroBias/RECO/09Oct2017-v1/00000/E8A1FC8A-0CAD-E711-8E5A-FA163E32849F.root'
        
        'file:/eos/cms/store/data/Run2017B/ZeroBias7/RECO/05Jul2017-v2/70000/04AB5407-C861-E711-B075-0CC47A6C1866.root',
        'file:/eos/cms/store/data/Run2017B/ZeroBias7/RECO/05Jul2017-v2/70000/3A0E9013-DD61-E711-A146-0CC47A78A41C.root'
       #'/store/relval/CMSSW_7_5_2/JetHT/MINIAOD/75X_dataRun1_HLT_frozen_v2_RelVal_jet2012D-v1/00000/7CEB618B-8151-E511-8D05-002618943857.root',
       #'/store/relval/CMSSW_7_5_2/JetHT/MINIAOD/75X_dataRun1_HLT_frozen_v2_RelVal_jet2012D-v1/00000/8A6ED13D-8351-E511-A6E1-0025905964C2.root',
       #'/store/relval/CMSSW_7_5_2/JetHT/MINIAOD/75X_dataRun1_HLT_frozen_v2_RelVal_jet2012D-v1/00000/9A6F45A5-8251-E511-8BB5-0025905964A6.root',
       #'/store/relval/CMSSW_7_5_2/JetHT/MINIAOD/75X_dataRun1_HLT_frozen_v2_RelVal_jet2012D-v1/00000/D6536366-7E51-E511-BC81-0025905A48F2.root',
       #'/store/relval/CMSSW_7_5_2/JetHT/MINIAOD/75X_dataRun1_HLT_frozen_v2_RelVal_jet2012D-v1/00000/DE6F609B-8251-E511-940D-002618943916.root' 
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
