import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "GR09_31X_V5P::All"
process.prefer("GlobalTag")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

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
    "rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_1.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_10.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_12.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_13.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_15.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_16.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_18.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_19.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_2.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_20.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_21.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_22.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_23.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_25.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_3.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_4.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_6.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_8.root",
"rfio:/castor/cern.ch/user/c/cerminar/DT-DQM/test/run109815/DTDQMOffline_9.root"
)
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

process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/Cosmics/CMSSW_2_2_X-Testing/RECO'

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
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        DTBlockedROChannelsTest = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(False),
        threshold = cms.untracked.string('WARNING')
    ),
    debugModules = cms.untracked.vstring('*')
)





