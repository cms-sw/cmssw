import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# CRAFT08
# process.GlobalTag.globaltag = CRAFT0831X_V1::All
# CRAFT09
process.GlobalTag.globaltag = "GR09_P_V2::All"
#process.GlobalTag.globaltag = "CRAFT09_R_V2::All"


process.load("DQM.SiStripMonitorClient.SiStripClientConfig_Tier0_cff")
process.siStripDaqInfo = cms.EDAnalyzer("SiStripDaqInfo")
process.siStripDcsInfo = cms.EDAnalyzer("SiStripDcsInfo")
process.siStripCertificationInfo = cms.EDAnalyzer("SiStripCertificationInfo")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.options = cms.untracked.PSet(
 fileMode = cms.untracked.string('FULLMERGE')
)

process.source = cms.Source("PoolSource",
#    dropMetaData = cms.untracked.bool(True),
    processingMode = cms.untracked.string("RunsLumisAndEvents"),
    fileNames = cms.untracked.vstring('file:sistrip_reco1.root', 'file:sistrip_reco2.root')
)

process.maxEvents.input = -1

process.source.processingMode = "RunsAndLumis"

process.DQMStore.referenceFileName = ''
process.dqmSaver.convention = 'Online'
process.dqmEnvTr = cms.EDAnalyzer("DQMEventInfo",
                                                 subSystemFolder = cms.untracked.string('Tracking'),
                                                 eventRateWindow = cms.untracked.double(0.5),
                                                 eventInfoFolder = cms.untracked.string('EventInfo')
                                )

#process.dqmSaver.workflow = '/GlobalCruzet4-A/CMSSW_2_1_X-Testing/RECO'

process.DQMStore.collateHistograms = False
process.EDMtoMEConverter.convertOnEndLumi = True
process.EDMtoMEConverter.convertOnEndRun = False

# DQM Utility to calculate # of bins
process.load("DQMServices.Components.DQMStoreStats_cfi")

# Tracer service
process.Tracer = cms.Service('Tracer',indentation = cms.untracked.string('$$'))
#process.load('DQM.SiStripCommon.MessageLogger_cfi')

process.p1 = cms.Path(process.EDMtoMEConverter*process.SiStripOfflineDQMClient*process.siStripDaqInfo*process.siStripDaqInfo*process.siStripDcsInfo*process.siStripCertificationInfo*process.dqmEnvTr*process.dqmSaver*process.dqmStoreStats)
