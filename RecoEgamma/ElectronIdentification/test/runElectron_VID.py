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
    # AOD test files from /store/relval/CMSSW_7_5_2/RelValZEE_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1    
    '/store/relval/CMSSW_7_5_2/RelValZEE_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/1A6B8B4F-9F50-E511-BF72-002354EF3BDC.root',
    '/store/relval/CMSSW_7_5_2/RelValZEE_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/140FBD4D-9F50-E511-8E4E-0025905A613C.root',
    '/store/relval/CMSSW_7_5_2/RelValZEE_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/428EF54E-9F50-E511-892C-0026189438E4.root',
    '/store/relval/CMSSW_7_5_2/RelValZEE_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/7A12234C-9F50-E511-B160-002618943986.root',
    '/store/relval/CMSSW_7_5_2/RelValZEE_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/9C439950-9F50-E511-93D8-00261894398B.root',
    '/store/relval/CMSSW_7_5_2/RelValZEE_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/E4438E4F-9F50-E511-9800-002354EF3BDC.root'    
    )    

inputFilesMiniAOD = cms.untracked.vstring(
    # MiniAOD test files from /store/relval/CMSSW_7_5_2/RelValZEE_13/MINIAODSIM/PU25ns_75X_mcRun2_asymptotic_v5-v1
    '/store/relval/CMSSW_7_5_2/RelValZEE_13/MINIAODSIM/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/5898DE4A-9F50-E511-8E2B-0026189438AF.root',
    '/store/relval/CMSSW_7_5_2/RelValZEE_13/MINIAODSIM/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/6CCF8C4C-9F50-E511-A3AF-0025905A48D0.root'
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
