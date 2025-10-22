import FWCore.ParameterSet.Config as cms

process = cms.Process('ParticleFlowDQMOffline')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

# load jet correctors
process.load('JetMETCorrections.Configuration.JetCorrectors_cff')

# my analyzer
process.load('DQMOffline.ParticleFlow.runBasic_cfi')


#
# This GT for MC should have "JetCorrectorParametersCollection_Winter25_AK4PFPuppi_offline_v1"
# https://cms-talk.web.cern.ch/t/gt-mc-gt-request-for-winter25-miniv6-nanov15-reprocessing-in-150x-for-2025-data-monitoring-for-jme-and-btv/124962
# https://cms-conddb.cern.ch/cmsDbBrowser/list/Prod/gts/150X_mcRun3_2025_realistic_v4
#
# GT = "150X_mcRun3_2025_realistic_v4"
# TAG_JEC = ""
# inputFile="/store/mc/Run3Winter25Reco/QCD_Bin-PT-15to7000_Par-PT-flat2022_TuneCP5_13p6TeV_pythia8/AODSIM/142X_mcRun3_2025_realistic_v9-v4/2810001/6a556fcb-beac-4ef4-9ed3-95df061cf1b1.root"
# outputFile="OUT_step1_MC.root"
# goldenJSONPath="" # Keep empty. Do not use for MC

#
# Use Prompt GT for Data then we manually replace the corrections in Prompt GT.
#
GT = "150X_dataRun3_Prompt_v1"
TAG_JEC = "JetCorrectorParametersCollection_Winter25Prompt25_RunC_V1_DATA_AK4PFPuppi_v1"
inputFile="/store/data/Run2025E/JetMET0/AOD/PromptReco-v1/000/395/987/00000/68ce8c81-1fc0-48a6-b71f-d2c93c1d3787.root"
outputFile="OUT_step1_Data.root"
goldenJSONPath="/eos/user/c/cmsdqm/www/CAF/certification/Collisions25/Cert_Collisions2025_391658_397294_Golden.json"

#
# Setup Global Tag
#
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, GT, '')

#
# Here we explicitly override the corrections in a Global Tag
#
if TAG_JEC != "":
    process.GlobalTag.toGet = cms.VPSet(
      cms.PSet(
        record = cms.string("JetCorrectionsRecord"),
        tag = cms.string(TAG_JEC),
        label = cms.untracked.string('AK4PFPuppi'),
        connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
      )
    )

#
# "CorrectedPFJetProducer" module applies the jet energy corrections
# on the jet collection and sort the collection according to pt
# https://cmssdt.cern.ch/lxr/source/JetMETCorrections/Modules/plugins/CorrectedJetProducers.cc#0014
# https://cmssdt.cern.ch/lxr/source/JetMETCorrections/Modules/interface/CorrectedJetProducer.h
#
process.ak4PFJetsPuppiCorrected = cms.EDProducer('CorrectedPFJetProducer',
    src        = cms.InputTag('ak4PFJetsPuppi'),
    correctors = cms.VInputTag('ak4PFPuppiL1FastL2L3ResidualCorrector')
)

from DQMOffline.ParticleFlow.runBasic_cfi import *

# back to original script
# with open('fileList_2.log') as f:
#     lines = f.readlines()

#Input source
process.source = cms.Source("PoolSource",
    # fileNames = cms.untracked.vstring(lines),
    fileNames = cms.untracked.vstring(inputFile)
)
###################################################################
# GoldenJSON Filtering
###################################################################
if goldenJSONPath != "":
    import FWCore.PythonUtilities.LumiList as LumiList
    process.source.lumisToProcess = LumiList.LumiList(filename = goldenJSONPath).getVLuminosityBlockRange()

from DQMOffline.ParticleFlow.runBasic_cfi import *

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string(outputFile)
)

process.p = cms.Path(
    process.ak4PFPuppiL1FastL2L3ResidualCorrectorChain+
    process.ak4PFJetsPuppiCorrected+
    process.PFAnalyzer
)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

## Schedule definition
process.schedule = cms.Schedule(
    process.p,
    process.DQMoutput_step
)


