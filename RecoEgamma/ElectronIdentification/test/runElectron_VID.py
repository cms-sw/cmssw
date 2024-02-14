
import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("TestElectrons")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
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
#AOD test files dataset :/RelValZEE_13/CMSSW_8_0_21-PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6-v1/GEN-SIM-RECO
inputFilesAOD = cms.untracked.vstring(
    '/store/relval/CMSSW_8_0_21/RelValZEE_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6-v1/10000/105EBFE7-9198-E611-8604-0025905B85D2.root',
'/store/relval/CMSSW_8_0_21/RelValZEE_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6-v1/10000/483FD411-9498-E611-9427-0025905B85D2.root',
'/store/relval/CMSSW_8_0_21/RelValZEE_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6-v1/10000/4ACA8789-A498-E611-A54C-0025905B858E.root',
'/store/relval/CMSSW_8_0_21/RelValZEE_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6-v1/10000/5A20FA98-9498-E611-9DB0-0CC47A78A33E.root',
'/store/relval/CMSSW_8_0_21/RelValZEE_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6-v1/10000/6694D992-9098-E611-8461-0CC47A4D768E.root',
'/store/relval/CMSSW_8_0_21/RelValZEE_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6-v1/10000/7887DD0D-9498-E611-8D98-0CC47A4C8E64.root',
'/store/relval/CMSSW_8_0_21/RelValZEE_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6-v1/10000/845B6ABD-9098-E611-A2E3-0CC47A78A4BA.root',
'/store/relval/CMSSW_8_0_21/RelValZEE_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6-v1/10000/A8877C9C-A498-E611-8EFF-0025905A607E.root',


    #'file:/opt/ppd/scratch/harper/mcTestFiles/ZToEE_NNPDF30_13TeV-powheg_M_200_400_PUSpring16_80X_mcRun2_asymptotic_2016_v3-v2_FE668C0A-8B09-E611-ACDE-B083FED76DBD.root'
    )    
#AOD test files dataset :/RelValZEE_13/CMSSW_8_0_21-PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6-v1/MINIAODSIM
inputFilesMiniAOD = cms.untracked.vstring(  
    '/store/relval/CMSSW_8_0_21/RelValZEE_13/MINIAODSIM/PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6-v1/10000/0E6B945E-A498-E611-BD29-0CC47A78A30E.root',
    '/store/relval/CMSSW_8_0_21/RelValZEE_13/MINIAODSIM/PU25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6-v1/10000/EE70625F-A498-E611-8CC7-0CC47A4D766C.root',
    )

# Set up input/output depending on the format
# You can list here either AOD or miniAOD files, but not both types mixed
#

print(sys.argv[1])
useAOD = bool(int(sys.argv[1]))

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
my_id_modules = [sys.argv[2]]

#add them to the VID producer
for idmod in my_id_modules:
    setupAllVIDIdsInModule(process,idmod,setupVIDElectronSelection)

# Make sure to add the ID sequence upstream from the user analysis module
process.p = cms.Path(process.egmGsfElectronIDSequence)
