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

# Setup Global Tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '150X_dataRun3_Prompt_v1', '')

# Here we explicitly override the jet energy corrections (JECs) in a Global Tag
process.GlobalTag.toGet = cms.VPSet(
  cms.PSet(
    record = cms.string("JetCorrectionsRecord"),
    tag = cms.string("JetCorrectorParametersCollection_Winter25Prompt25_RunC_V1_DATA_AK4PFPuppi_v1"),
    label = cms.untracked.string('AK4PFPuppi'),
    connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
  )
)

with open('fileList.log') as f:
    lines = f.readlines()
#Input source
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(lines))

# "CorrectedPFJetProducer" module applies the jet energy
# corrections on the jet collection and sort the collection
# according to pt
process.ak4PFJetsPuppiCorrected = cms.EDProducer('CorrectedPFJetProducer',
    src        = cms.InputTag('ak4PFJetsPuppi'),
    correctors = cms.VInputTag('ak4PFPuppiL1FastL2L3ResidualCorrector')
)

###################################################################
# Data certification GoldenJSON filtering
###################################################################
goldenJSONPath="/eos/user/c/cmsdqm/www/CAF/certification/Collisions25/Cert_Collisions2025_391658_397294_Golden.json"
if goldenJSONPath != "":
    import FWCore.PythonUtilities.LumiList as LumiList
    process.source.lumisToProcess = LumiList.LumiList(filename = goldenJSONPath).getVLuminosityBlockRange()

from DQMOffline.ParticleFlow.runBasic_cfi import *

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
                                     fileName = cms.untracked.string("OUT_step1.root"))


process.p = cms.Path(
    process.ak4PFPuppiL1FastL2L3ResidualCorrectorChain+
    process.ak4PFJetsPuppiCorrected+
    process.PFAnalyzer)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)


## Schedule definition
process.schedule = cms.Schedule(
    process.p,
    process.DQMoutput_step
    )









