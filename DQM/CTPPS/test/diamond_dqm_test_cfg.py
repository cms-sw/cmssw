import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('RECODQM', Run3)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )
process.verbosity = cms.untracked.PSet( input = cms.untracked.int32(10) ) 

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ) 
)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# load DQM framework
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "CTPPS"
process.dqmEnv.eventInfoFolder = "EventInfo"
process.dqmSaver.path = ""
process.dqmSaver.tag = "CTPPS"
#process.dqmSaver.runNumber = 999999


# raw data source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#'file:/eos/cms/store/data/Run2022B/AlCaPPS/RAW/v1/000/355/207/00000/c23440f4-49c0-44aa-b8f6-f40598fb4705.root',
#'file:/eos/cms/store/group/dpg_ctpps/comm_ctpps/ALCAPPS-RAW/000/355/988/00000/7728d214-b404-4578-b781-b0c207cfb875.root',
#'/store/data/Run2022D/AlCaPPSPrompt/ALCARECO/PPSCalMaxTracks-PromptReco-v2/000/357/900/00000/0434a2eb-2cea-4d83-8b21-fe91b755e62e.root',
#'file:/eos/cms/store/express/Run2022D/StreamALCAPPSExpress/ALCARECO/PPSCalMaxTracks-Express-v2/000/357/900/00000/4e3126cd-3dae-48b2-a2e5-b14542500030.root'
'/store/data/Run2022B/AlCaPPS/RAW/v1/000/355/207/00000/c23440f4-49c0-44aa-b8f6-f40598fb4705.root'
    ),
)



from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_relval', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '124X_dataRun3_Prompt_frozen_v5', '')
process.GlobalTag.toGet = cms.VPSet()
process.GlobalTag.toGet.append(
  cms.PSet(record = cms.string("PPSTimingCalibrationRcd"),
           tag =  cms.string("PPSDiamondTimingCalibration_Run3_recovered_v3"),
           label = cms.untracked.string('PPSTestCalibration'),
           connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS")
	)
)

# process.GlobalTag.toGet.append(
#     cms.PSet(
#         record=cms.string('TotemReadoutRcd'),
#         tag=cms.string('DiamondDAQMapping'),
#         label=cms.untracked.string('TimingDiamond'),
#         connect=cms.string('sqlite_file:CTPPSDiamond_DAQMapping.db')
#     )
# )

# process.GlobalTag.toGet.append(
#     cms.PSet(
#         record=cms.string('TotemAnalysisMaskRcd'),
#         tag=cms.string('DiamondDAQMapping1'),
#         label=cms.untracked.string('TimingDiamond'),
#         connect=cms.string('sqlite_file:CTPPSDiamond_DAQMapping.db')
#     )
# )


# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

process.ctppsPixelDigis.inputLabel = cms.InputTag("hltPPSCalibrationRaw")
process.ctppsDiamondRawToDigi.rawDataTag = cms.InputTag("hltPPSCalibrationRaw")
process.totemRPRawToDigi.rawDataTag = cms.InputTag("hltPPSCalibrationRaw")
process.totemTimingRawToDigi.rawDataTag = cms.InputTag("hltPPSCalibrationRaw")


# local RP reconstruction chain with standard settings
process.load("RecoPPS.Configuration.recoCTPPS_cff")
process.ctppsDiamondRecHits.digiTag=cms.InputTag('ctppsDiamondRawToDigi:TimingDiamond')
process.ctppsDiamondRecHits.timingCalibrationTag=cms.string("GlobalTag:PPSTestCalibration")

# process.ctppsDiamondRecHits.timingCalibrationTag=cms.string("")
# process.load('CondCore.CondDB.CondDB_cfi')
# process.CondDB.connect = 'sqlite_file:../../../CalibPPS/TimingCalibration/test/ppsDiamondTiming_calibration355207.sqlite' # SQLite input
# process.PoolDBESSource = cms.ESSource('PoolDBESSource',
#         process.CondDB,
#         DumpStats = cms.untracked.bool(True),
#         toGet = cms.VPSet(
#             cms.PSet(
#                 record = cms.string('PPSTimingCalibrationRcd'),
#                 tag = cms.string('DiamondTimingCalibration') 
#             )
#         )
#     )

#process.load('Geometry.VeryForwardGeometry.geometryRPFromDD_2021_cfi')
# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")
process.ctppsDiamondDQMSource.tagDigi=cms.untracked.InputTag('ctppsDiamondRawToDigi:TimingDiamond')
process.ctppsDiamondDQMSource.tagFEDInfo=cms.untracked.InputTag('ctppsDiamondRawToDigi:TimingDiamond')
process.ctppsDiamondDQMSource.tagStatus=cms.untracked.InputTag('ctppsDiamondRawToDigi:TimingDiamond')
process.ctppsDiamondDQMSource.tagPixelLocalTracks=cms.untracked.InputTag('ctppsPixelLocalTracks')
#process.ctppsDiamondDQMSource.excludeMultipleHits = cms.bool(True)
process.ctppsDiamondDQMSource.plotOnline = cms.untracked.bool(True)
process.ctppsDiamondDQMSource.plotOffline = cms.untracked.bool(False)

process.content = cms.EDAnalyzer("EventContentAnalyzer")

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.tdcCalibShiftDQM = DQMEDAnalyzer('TdcCalibShiftDQM')
process.path = cms.Path(
    #process.content*
    process.ctppsRawToDigi*
    process.recoCTPPS*
    process.ctppsDQMOnlineSource*
    ##process.ctppsDiamondDQMSource*
    process.ctppsDQMOnlineHarvest
    #process.tdcCalibShiftDQM
    

    #process.ctppsDiamondRecHits *
    #process.ctppsDiamondLocalTracks *
    #process.ctppsLocalTrackLiteProducer *
    #process.ctppsProtons *
    #process.ctppsDiamondDQMSource*
    #process.ctppsDQMOnlineHarvest
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("file:test.root"),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_ctpps*_*_*',
    ),
)


process.end_path = cms.EndPath(
    #process.output
    process.dqmEnv +
    process.dqmSaver
)

process.schedule = cms.Schedule(
    process.path,
    process.end_path
) 

#process.output = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string("AOD.root"),
#    outputCommands = cms.untracked.vstring(
#        'drop *',
#        'keep *_ctpps*_*_*',
#    ),
#)  
#process.outpath = cms.EndPath(process.output)
