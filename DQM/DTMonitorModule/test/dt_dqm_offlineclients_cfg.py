import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "GR09_31X_V5P::All"
process.prefer("GlobalTag")

process.load("Configuration.StandardSequences.Geometry_cff")

#process.load("DQMOffline.Configuration.DQMOfflineCosmics_SecondStep_cff")
process.load("DQM.DTMonitorClient.dtDQMOfflineClients_cff")
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
    fileNames = cms.untracked.vstring(
"rfio:/castor/cern.ch//user/c/cerminar/DT-DQM/test/run109459/DTDQMOffline_1.root",
                            "rfio:/castor/cern.ch//user/c/cerminar/DT-DQM/test/run109459/DTDQMOffline_10.root",
                            "rfio:/castor/cern.ch//user/c/cerminar/DT-DQM/test/run109459/DTDQMOffline_2.root",
                            "rfio:/castor/cern.ch//user/c/cerminar/DT-DQM/test/run109459/DTDQMOffline_3.root",
                            "rfio:/castor/cern.ch//user/c/cerminar/DT-DQM/test/run109459/DTDQMOffline_4.root",
                            "rfio:/castor/cern.ch//user/c/cerminar/DT-DQM/test/run109459/DTDQMOffline_5.root",
                            "rfio:/castor/cern.ch//user/c/cerminar/DT-DQM/test/run109459/DTDQMOffline_6.root",
                            "rfio:/castor/cern.ch//user/c/cerminar/DT-DQM/test/run109459/DTDQMOffline_7.root",
                            "rfio:/castor/cern.ch//user/c/cerminar/DT-DQM/test/run109459/DTDQMOffline_8.root",
                            "rfio:/castor/cern.ch//user/c/cerminar/DT-DQM/test/run109459/DTDQMOffline_9.root"
                            )
)

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
                      process.dtClients*
                      process.dtCertification*
                      process.dqmSaver*
                      process.dqmStoreStats)


# message logger
process.MessageLogger = cms.Service("MessageLogger",
                                    debugModules = cms.untracked.vstring('*'),
                                    destinations = cms.untracked.vstring('cout'),
                                    categories = cms.untracked.vstring('DTTimeEvolutionHisto'), 
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'),
                                                              noLineBreaks = cms.untracked.bool(False),
                                                              DEBUG = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(0)),
                                                              INFO = cms.untracked.PSet(
                                                                      limit = cms.untracked.int32(0)),
                                                              DTTimeEvolutionHisto = cms.untracked.PSet(
                                                                                 limit = cms.untracked.int32(-1))
                                                              )
                                    )





