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
    # AOD test files from /store/relval/CMSSW_7_6_0_pre4/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1
    '/store/relval/CMSSW_7_6_0_pre4/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/0E37A324-714F-E511-B658-003048FFD770.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/18C42D26-714F-E511-90A9-0025905B855C.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/2448F11B-734F-E511-99C8-0025905A608E.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/3A648F28-E64F-E511-9291-0026189438E1.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/5211471B-724F-E511-9B00-0025905A613C.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/8C6C961C-734F-E511-92F5-003048FF9AC6.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/BC2BE168-E74F-E511-B126-0025905A60B0.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/BEA93A19-724F-E511-B0C0-0025905A60F4.root'
    )    

inputFilesMiniAOD = cms.untracked.vstring(
    # MiniAOD test files from /store/relval/CMSSW_7_6_0_pre4/RelValH130GGgluonfusion_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1
    '/store/relval/CMSSW_7_6_0_pre4/RelValH130GGgluonfusion_13/MINIAODSIM/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/14954967-E74F-E511-BEF2-0026189438EF.root',
    '/store/relval/CMSSW_7_6_0_pre4/RelValH130GGgluonfusion_13/MINIAODSIM/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/2061DB66-E74F-E511-9531-0026189438DB.root'
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
