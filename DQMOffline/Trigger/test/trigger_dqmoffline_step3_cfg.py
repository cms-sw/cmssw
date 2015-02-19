import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")

process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
#
#  DQMOffline
#
process.load("DQMOffline.Configuration.DQMOffline_SecondStep_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
 fileMode = cms.untracked.string('FULLMERGE')
)

process.source = cms.Source("PoolSource",
#    dropMetaData = cms.untracked.bool(True),
    processingMode = cms.untracked.string("RunsLumisAndEvents"),
    fileNames = cms.untracked.vstring(
      'file:myDQMOfflineTriggerEDM.root'
    )
)

process.source.processingMode = "RunsAndLumis"

process.DQMStore.referenceFileName = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/StreamExpress/BeamCommissioning09-v8/DQMOffline'

process.DQMStore.collateHistograms = False
process.EDMtoMEConverter.convertOnEndLumi = True
process.EDMtoMEConverter.convertOnEndRun = False

process.p1 = cms.Path(process.EDMtoMEConverter*process.triggerOfflineDQMClient * process.hltOfflineDQMClient * process.dqmSaver)
#process.p1 = cms.Path(process.EDMtoMEConverter*process.dqmSaver)

