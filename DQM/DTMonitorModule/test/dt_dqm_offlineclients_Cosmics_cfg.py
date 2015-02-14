import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "GR09_31X_V5P::All"
process.prefer("GlobalTag")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#process.load("DQMOffline.Configuration.DQMOfflineCosmics_SecondStep_cff")
process.load("DQM.DTMonitorClient.dtDQMOfflineClients_Cosmics_cff")
process.load("DQM.DTMonitorClient.dtDQMOfflineCertification_cff")
#process.load("DQMOffline.Configuration.DQMOfflineCosmics_Certification_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.options = cms.untracked.PSet(
 fileMode = cms.untracked.string('FULLMERGE')
)

process.source = cms.Source("PoolSource",
                            #    dropMetaData = cms.untracked.bool(True),
                            processingMode = cms.untracked.string("RunsLumisAndEvents"),
                            fileNames = cms.untracked.vstring('file:DTDQMOffline.root')
)


# process.source = cms.Source("PoolSource",
# #    dropMetaData = cms.untracked.bool(True),
#     processingMode = cms.untracked.string("RunsLumisAndEvents"),
#     fileNames = cms.untracked.vstring(
#     "file:DTDQMOffline_1.root",
#     "file:DTDQMOffline_2.root",
#     "file:DTDQMOffline_3.root")
# )


process.maxEvents.input = -1

process.source.processingMode = "RunsAndLumis"

process.DQMStore.referenceFileName = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/Cosmics/CMSSW_2_2_X-Testing/RECO'

process.DQMStore.collateHistograms = False
process.EDMtoMEConverter.convertOnEndLumi = True
process.EDMtoMEConverter.convertOnEndRun = False

process.load("DQMServices.Components.DQMStoreStats_cfi")

process.p1 = cms.Path(process.EDMtoMEConverter*
                      process.dtClientsCosmics*
                      process.dtCertification*
                      process.dqmSaver*
                      process.dqmStoreStats)


# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring('*'),
                                    destinations = cms.untracked.vstring('cout'),
                                    categories = cms.untracked.vstring('DTBlockedROChannelsTest'), 
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'),
                                                              noLineBreaks = cms.untracked.bool(False),
                                                              DEBUG = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(0)),
                                                              INFO = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(0)),
                                                              DTBlockedROChannelsTest = cms.untracked.PSet(
                                                                                 limit = cms.untracked.int32(-1))
                                                              )
                                    )





