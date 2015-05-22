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
process.GlobalTag.globaltag ='MCRUN2_74_V9'

# check # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/2AE21D5C-6FF1-E411-B3AC-02163E00E60F.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/2C0E0382-68F1-E411-A8B2-02163E00E640.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/34FEEE59-70F1-E411-9554-02163E00F420.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/403D5922-84F1-E411-8F42-02163E00AD2E.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/42658BC2-69F1-E411-9D7E-02163E00F8B3.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/4E24BEC6-8FF1-E411-92A5-02163E010FCF.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/5CB03F23-5EF1-E411-9294-02163E013B50.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/6A08C46A-6DF1-E411-93A5-02163E00F298.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/6E82B231-7BF1-E411-BBD7-02163E013D2D.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/76E26B39-61F1-E411-BB03-02163E00E814.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/84E5227D-73F1-E411-B9F9-02163E010FD9.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/8C966E42-72F1-E411-95F8-02163E0130C3.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/9A1C1859-63F2-E411-8935-02163E00E913.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/A65D6BC7-64F1-E411-B8E9-02163E00F710.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/A6FF4CF5-6CF1-E411-9580-02163E00E61F.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/A8E1054B-EDF1-E411-864F-02163E00B782.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/CECD5CE1-66F1-E411-BD44-02163E0130F9.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/D27937E4-6AF1-E411-B9E8-02163E00C323.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/E6AD30C8-63F1-E411-9828-02163E00B999.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/F0A1C285-75F1-E411-9981-02163E00E96D.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/F4B1EF05-69F1-E411-BFEF-02163E00EBA7.root',
       '/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/MCRUN2_74_V9_extended-v2/00000/FC2233E0-72F1-E411-82EA-02163E00E5B6.root'
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
