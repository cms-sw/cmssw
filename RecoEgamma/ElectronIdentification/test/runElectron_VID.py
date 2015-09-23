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
    # AOD test files from /store/relval/CMSSW_7_4_8_patch1/RelValZEE_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1
    '/store/relval/CMSSW_7_4_8_patch1/RelValZEE_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/001C7DBD-583C-E511-9107-0025905A60B2.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValZEE_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/124BE41A-493C-E511-B909-0026189438DF.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValZEE_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/AC6B87BB-953C-E511-8A84-002618943918.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValZEE_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/C438BB08-5D3C-E511-9726-0025905A6084.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValZEE_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/C89CDBD4-443C-E511-B817-0026189438A2.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValZEE_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/F838C1BD-953C-E511-A59C-0025905A7786.root'
    )    

inputFilesMiniAOD = cms.untracked.vstring(
    # MiniAOD test files from /store/relval/CMSSW_7_4_8_patch1/RelValZEE_13/MINIAODSIM/PU25ns_MCRUN2_74_V11_mulTrh-v1
    '/store/relval/CMSSW_7_4_8_patch1/RelValZEE_13/MINIAODSIM/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/445E75C0-953C-E511-95A1-0025905A6056.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValZEE_13/MINIAODSIM/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/D85014C3-953C-E511-B0C0-003048FFD71E.root'
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
