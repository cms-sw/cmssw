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
    # AOD test files from /store/relval/CMSSW_7_5_2/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1
    '/store/relval/CMSSW_7_5_2/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/26521749-9F50-E511-9B5E-00261894390E.root',
    '/store/relval/CMSSW_7_5_2/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/86871F4C-9F50-E511-8310-0026189438A7.root',
    '/store/relval/CMSSW_7_5_2/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/90949C49-9F50-E511-A621-0026189438A9.root',
    '/store/relval/CMSSW_7_5_2/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/A655744A-9F50-E511-A921-002618943865.root',
    '/store/relval/CMSSW_7_5_2/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/B64A7D4B-9F50-E511-B3A1-0025905A60EE.root',
    '/store/relval/CMSSW_7_5_2/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/CE6D704A-9F50-E511-A825-002618943865.root',
    '/store/relval/CMSSW_7_5_2/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/E8DD7E48-9F50-E511-B689-002618943924.root'
    )    

inputFilesMiniAOD = cms.untracked.vstring(
    # MiniAOD test files from /store/relval/CMSSW_7_5_2/RelValH130GGgluonfusion_13/MINIAODSIM/PU25ns_75X_mcRun2_asymptotic_v5-v1
    '/store/relval/CMSSW_7_5_2/RelValH130GGgluonfusion_13/MINIAODSIM/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/4C6306C3-9F50-E511-A7DD-0025905A608C.root',
    '/store/relval/CMSSW_7_5_2/RelValH130GGgluonfusion_13/MINIAODSIM/PU25ns_75X_mcRun2_asymptotic_v5-v1/00000/56A009C4-9F50-E511-A998-002590596490.root'
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
