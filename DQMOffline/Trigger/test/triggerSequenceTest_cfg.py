from __future__ import print_function
import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
process.load("DQMServices.Core.DQM_cfg")

#process.load("DQM.HLTEvF.HLTMonitor_cff")
# remove because the client stuff is useless
#process.load("DQM.HLTEvF.HLTMonitorClient_cff")

# Don't load everything
#process.load("DQMOffline.Trigger.DQMOffline_Trigger_cff")

# Only load this
process.load("DQMOffline.Trigger.HLTGeneralOffline_cfi")

# remove this because we don't want to do local reconstruction
#process.load("Configuration.StandardSequences.GeometryPilot2_cff")
#process.load("Configuration.StandardSequences.MagneticField_cff")

process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )

# use this tag for 52
#process.GlobalTag.globaltag = 'GR_R_52_V4::All'

# use this tag for 42
print("\n\n\n------ WARNING USING GLOBAL TAG FOR 42X------\n\n\n\n")
process.GlobalTag.globaltag = 'GR_R_42_V22::All'

process.prefer("GlobalTag")

# removed this because we don't want to do local reco
#SiStrip Local Reco
#process.load("CalibTracker.SiStripCommon.TkDetMapESProducer_cfi")

#process.GlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" )

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

#process.source = cms.Source("PoolSource",
##    fileNames = cms.untracked.vstring('/store/data/Commissioning08/Monitor/RAW/v1/000/068/021/000BDAE4-37A6-DD11-8411-001D09F242EF.root')
#    #fileNames = cms.untracked.vstring('file:/tmp/hdyoo/8809F7DE-BFCA-DE11-B492-001617E30D40.root')
#    #fileNames = cms.untracked.vstring('/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/119/824/8809F7DE-BFCA-DE11-B492-001617E30D40.root')
#    fileNames = cms.untracked.vstring('/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/122/314/0C7140FD-7CD8-DE11-A515-001D09F2960F.root')#
# )


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    # never use old files to do this test
    # you will be sad
#	 'file:/data/ndpc0/c/abrinke1/RAW/170354/SingleMu/08B0697A-B7B0-E011-B0DE-003048D375AA.root'
    '/store/data/Run2011B/SingleMu/AOD/PromptReco-v1/000/180/241/C0F4F7A3-EF04-E111-A94F-003048D2C0F0.root'
    #'/store/relval/CMSSW_5_2_0/RelValTTbar/GEN-SIM-RECO/START52_V4A-v1/0248/8698CFBB-1869-E111-8121-00304867C1BA.root'
    
    )
)


process.p = cms.EndPath(process.dqmHLTFiltersDQMonitor)
  
process.pp = cms.Path(process.dqmEnv+process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = 'srv-c2d05-12'
process.DQM.collectorPort = 9190
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'HLT'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

# # Message Logger
# process.load("FWCore.MessageService.MessageLogger_cfi")
# #
# process.MessageLogger.destinations = ['cout', 'detailedInfo', 'critical']
# process.MessageLogger.cout = cms.untracked.PSet(
#     #threshold = cms.untracked.string('ERROR'),
#     #threshold = cms.untracked.string('INFO'),
#     #INFO = cms.untracked.PSet(
#     # limit = cms.untracked.int32(-1)
#     #)#,
#     threshold = cms.untracked.string('DEBUG'),
#     DEBUG = cms.untracked.PSet(
#     limit = cms.untracked.int32(-1) ## DEBUG, all messages
#     )
#     )
# 
# # copy stdout to a file
# process.MessageLogger.detailedInfo = process.MessageLogger.cout
# process.MessageLogger.debugModules = ['dqmHLTFiltersDQMonitor']
# process.MessageLogger.critical = cms.untracked.PSet(
#     threshold = cms.untracked.string('ERROR'),
#     #threshold = cms.untracked.string('INFO'),
#     #INFO = cms.untracked.PSet(
#     # limit = cms.untracked.int32(-1)
#     #)#,
#     #threshold = cms.untracked.string('DEBUG'),
#     ERROR = cms.untracked.PSet(
#     limit = cms.untracked.int32(-1) ## all messages
#     )
#     )
# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )



outfile = open('config.py','w')
print(process.dumpPython(), file=outfile)
outfile.close()

																						           
