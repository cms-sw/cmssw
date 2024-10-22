import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

#
#  DQM SERVICES
#
process.load("DQMServices.Core.DQM_cfg")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#
#  ME to EDM 
#
process.load("DQMServices.Components.MEtoEDMConverter_cff")
#
# DQMOffline
#
#process.load("DQMOffline.Configuration.DQMOfflineCosmics_cff")
process.load("DQMOffline.Configuration.DQMOffline_cff")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.load("curRun_files_cfi")

#process.source = cms.Source("PoolSource",
#    fileNames = 
#        cms.untracked.vstring(
##'/store/data/BeamCommissioning09/Cosmics/RECO/v1/000/120/020/DE706F2B-5FCC-DE11-98C7-001617E30CC8.root',
#
#				)
#)

process.EDM = cms.OutputModule("PoolOutputModule",
#process.RECOEventContent,
#dataset = cms.untracked.PSet(dataTier = cms.untracked.string('RECO')),
    fileName = cms.untracked.string('myDQMOfflineTriggerEDM.root'),
    outputCommands = cms.untracked.vstring('drop *', 
         'keep  *_*_*_DQM')
)


process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))

#process.MessageLogger = cms.Service("MessageLogger",
#    detailedInfo= cms.untracked.PSet(
#      threshold = cms.untracked.string('DEBUG'),
#      DEBUG = cms.untracked.PSet(
#            limit = cms.untracked.int32(10000)
#      )
#    ),
#    critical = cms.untracked.PSet(
#        threshold = cms.untracked.string('ERROR')
#    ),
#    debugModules = cms.untracked.vstring('dqmHLTFiltersDQMonitor'),
##debugModules = cms.untracked.vstring('*'),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('WARNING'),
#        WARNING = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#        ),
#        noLineBreaks = cms.untracked.bool(True)
#    ),
#    destinations = cms.untracked.vstring('detailedInfo', 
#        'critical', 
#        'cout')
##    destinations = cms.untracked.vstring( 'critical', 'cout')
#)
process.triggerOfflineDQMSource.remove(process.l1tgmt)
process.triggerOfflineDQMSource.remove(process.l1tcsctf)

process.AllPath = cms.Path(process.triggerOfflineDQMSource *  process.MEtoEDMConverter)
#process.AllPath = cms.Path(process.dqmHLTFiltersDQMonitor *  process.MEtoEDMConverter)

process.outpath = cms.EndPath(process.EDM)

