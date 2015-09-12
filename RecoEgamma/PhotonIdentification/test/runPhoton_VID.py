import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("TestPhotons")

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
    # AOD test files from /store/relval/CMSSW_7_4_8_patch1/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1
    '/store/relval/CMSSW_7_4_8_patch1/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/48581710-3D3C-E511-981F-0025905B85A2.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/58988779-3B3C-E511-8934-0025905A6070.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/5A198C10-403C-E511-AD13-003048FFD7C2.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/74FE2C8B-393C-E511-8208-0025905964CC.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/9EFB0388-393C-E511-81E5-002618943843.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/C28A140E-403C-E511-9458-0026189438EF.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/D85E6C7B-3B3C-E511-BD48-0025905A6090.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/E2A43911-3D3C-E511-9A46-0025905938AA.root'
    )    

inputFilesMiniAOD = cms.untracked.vstring(
    # MiniAOD test files from /store/relval/CMSSW_7_4_8_patch1/RelValH130GGgluonfusion_13/MINIAODSIM/PU25ns_MCRUN2_74_V11_mulTrh-v1
    '/store/relval/CMSSW_7_4_8_patch1/RelValH130GGgluonfusion_13/MINIAODSIM/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/96F1561D-403C-E511-A102-002618943856.root',
    '/store/relval/CMSSW_7_4_8_patch1/RelValH130GGgluonfusion_13/MINIAODSIM/PU25ns_MCRUN2_74_V11_mulTrh-v1/00000/BE1C7C22-403C-E511-A986-0025905B858C.root'
    )

# Set up input/output depending on the format
# You can list here either AOD or miniAOD files, but not both types mixed
#

print sys.argv[2]
useAOD = bool(int(sys.argv[2]))

if useAOD == True :
    inputFiles = inputFilesAOD
    outputFile = "photon_ntuple.root"
    print("AOD input files are used")
else :
    inputFiles = inputFilesMiniAOD
    outputFile = "photon_ntuple_mini.root"
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

switchOnVIDPhotonIdProducer(process, dataFormat)

# define which IDs we want to produce
my_id_modules = [sys.argv[3]]

#add them to the VID producer
for idmod in my_id_modules:
    setupAllVIDIdsInModule(process,idmod,setupVIDPhotonSelection)

# Make sure to add the ID sequence upstream from the user analysis module
process.p = cms.Path(process.egmPhotonIDSequence)
