import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

#----------------------------
# DQM Environment
#-----------------------------
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#----------------------------
# BeamMonitor
#-----------------------------
process.load("DQM.BeamMonitor.BeamMonitor_Cosmics_cff") # need input track collection in the event
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff") # need beam spot collection in the event

####  SETUP TRACKING RECONSTRUCTION ####

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.GeometryPilot2_cff")

#-----------------------------
# Magnetic Field
#-----------------------------
#process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')

#--------------------------
# Calibration
#--------------------------
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG"
process.GlobalTag.globaltag = 'GR09_31X_V6H::All' # or any other appropriate
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#-----------------------
#  Reconstruction Modules
#-----------------------
# Real data raw to digi
process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")
process.siStripDigis.ProductLabel = 'source'
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = 'source'

# Local and Track Reconstruction
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff")
process.load("RecoTracker.Configuration.RecoTrackerP5_cff")

# offline beam spot
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

#### END OF TRACKING RECONSTRUCTION ####

#----------------------------
# Event Source
#-----------------------------
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(
	'file:/lookarea_SM/CRAFT2_2009.00112222.0001.A.storageManager.00.0000.dat',
	'file:/lookarea_SM/CRAFT2_2009.00112222.0006.A.storageManager.01.0000.dat',
	'file:/lookarea_SM/CRAFT2_2009.00112222.0011.A.storageManager.02.0000.dat',
	'file:/lookarea_SM/CRAFT2_2009.00112222.0016.A.storageManager.03.0000.dat',
	'file:/lookarea_SM/CRAFT2_2009.00112222.0021.A.storageManager.04.0000.dat',
	'file:/lookarea_SM/CRAFT2_2009.00112222.0026.A.storageManager.05.0000.dat',
	'file:/lookarea_SM/CRAFT2_2009.00112222.0031.A.storageManager.06.0000.dat'
    )
)

process.tracking = cms.Path(process.siPixelDigis*process.siStripDigis*process.offlineBeamSpot*process.trackerlocalreco*process.ctftracksP5*process.cosmictracksP5)
process.monitor = cms.Path(process.dqmBeamMonitor+process.dqmBeamCondMonitor+process.dqmEnv+process.dqmSaver)

process.DQMStore.verbose = 0
process.DQM.collectorHost = 'srv-c2d05-18'
process.DQM.collectorPort = 9090
process.dqmSaver.dirName = '.'
process.dqmSaver.producer = 'Playback'
process.dqmSaver.convention = 'Online'
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.saveByRun = 1
process.dqmSaver.saveAtJobEnd = True

# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

#process.out = cms.OutputModule("PoolOutputModule",
#                               fileName = cms.untracked.string('test.root'),
#			       outputCommands = cms.untracked.vstring(
#				'drop *',
#				'keep *_offlineBeamSpot_*_*',
#				'keep *_ctfWithMaterialTracksP5_*_*',
#				'keep *_cosmictrackfinderP5_*_*'
#			       )
#                              )
#process.end = cms.EndPath(process.out)

process.schedule = cms.Schedule(process.tracking, process.monitor)

