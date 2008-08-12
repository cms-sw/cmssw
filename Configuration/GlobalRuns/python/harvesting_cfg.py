import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
# Accumulation of globally transformed data
#
#module EDMtoMEConverter
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRZT210_V1P::All"
process.prefer("GlobalTag")

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

# execute path
#
process.load("DQMOffline.Configuration.DQMOffline_SecondStep_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
#    dropMetaData = cms.untracked.bool(True),
    processingMode = cms.untracked.string("RunsLumisAndEvents"),
    fileNames = cms.untracked.vstring('file:reco2.root')
)

process.maxEvents.input = -1

#process.source.processingMode = "Runs"
process.source.processingMode = "RunsAndLumis"

process.DQMStore.referenceFileName = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/GlobalCruzet3-A/CMSSW_2_1_0-Testing/RECO'

#process.DQMStore.collateHistograms = False
process.EDMtoMEConverter.convertOnEndLumi = True
process.EDMtoMEConverter.convertOnEndRun = False

process.p1 = cms.Path(process.EDMtoMEConverter*process.DQMOffline_SecondStep*process.dqmSaver)

