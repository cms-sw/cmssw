import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("TestElectrons")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load("Configuration.StandardSequences.GeometryDB_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# NOTE: the pick the right global tag!
#    for PHYS14 scenario PU4bx50 : global tag is ???
#    for PHYS14 scenario PU20bx25: global tag is PHYS14_25_V1
#  as a rule, find the global tag in the DAS under the Configs for given dataset
#process.GlobalTag.globaltag = 'PHYS14_25_V1::All'
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

#
# Define input data to read
#
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

inputFilesAOD = cms.untracked.vstring(
    # AOD test files from /store/relval/CMSSW_7_6_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1
    '/store/relval/CMSSW_7_6_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/3ADB5D32-DD4F-E511-AC01-002618943811.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/54B6CF34-DD4F-E511-9629-002590596490.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/8043D96A-6C4F-E511-81E7-003048FFD736.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/8E554BD2-6D4F-E511-BFD2-0025905A60DE.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/98EB5C3F-6D4F-E511-910B-0025905A6056.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/9C8CF66A-6C4F-E511-BD02-00259059391E.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/D015FB85-6C4F-E511-88FE-002618943902.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValZEE_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/D873CC62-6C4F-E511-ABBA-0025905B855E.root'
    )    

inputFilesMiniAOD = cms.untracked.vstring(
    # MiniAOD test files from /store/relval/CMSSW_7_6_0_pre4/RelValZEE_13/MINIAODSIM/PU25ns_76X_mcRun2_asymptotic_v1-v1
    '/store/relval/CMSSW_7_6_0_pre4/RelValZEE_13/MINIAODSIM/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/BE21962F-DD4F-E511-B681-002354EF3BDF.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValZEE_13/MINIAODSIM/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/D2B5E032-DD4F-E511-96A4-0025905A610C.root'
    )

# Set up input/output depending on the format
# You can list here either AOD or miniAOD files, but not both types mixed
#

print sys.argv[2]
useAOD = bool(int(sys.argv[2]))

if useAOD == True :
    inputFiles = inputFilesAOD
    outputFile = "electron_ntuple.root"
    print("AOD input files are used")
else :
    inputFiles = inputFilesMiniAOD
    outputFile = "electron_ntuple_mini.root"
    print("MiniAOD input files are used")
process.source = cms.Source ("PoolSource", fileNames = inputFiles )                             

#
# Set up electron ID (VID framework)
#

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
# turn on VID producer, indicate data format  to be
# DataFormat.AOD or DataFormat.MiniAOD, as appropriate 
if useAOD == True :
    dataFormat = DataFormat.AOD
else :
    dataFormat = DataFormat.MiniAOD

switchOnVIDElectronIdProducer(process, dataFormat)

# define which IDs we want to produce
my_id_modules = [sys.argv[3]]

#add them to the VID producer
for idmod in my_id_modules:
    setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection)

# Make sure to add the ID sequence upstream from the user analysis module
process.p = cms.Path(process.egmGsfElectronIDSequence)
