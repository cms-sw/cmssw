import FWCore.ParameterSet.Config as cms

process = cms.Process("harvester")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag = GlobalTag(process.GlobalTag, autoCond['run3_data_prompt'], '')
#process.GlobalTag.toGet.append(
#  cms.PSet(record = cms.string("PPSTimingCalibrationRcd"),
#           tag =  cms.string("PPSDiamondSampicCalibration_test"),
#           label = cms.untracked.string("DiamondSampicCalibration"),
#           connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
#	)
#)


process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")
process.load("RecoPPS.Configuration.recoCTPPS_cff")

# Source (histograms)
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring("file:worker_output.root"),
)

# output service for database
process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect = 'sqlite_file:ppsDiamondSampicTiming_calibration.sqlite' # SQLite output

process.PoolDBOutputService = cms.Service('PoolDBOutputService',
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('PPSTimingCalibrationRcd_SAMPIC'),
            tag = cms.string('PPSDiamondSampicCalibration_pcl'),
        )
    )
)


process.CondDB.connect = 'sqlite_file:corrected_sampic.sqlite' # SQLite input
process.PoolDBESSource = cms.ESSource('PoolDBESSource',
        process.CondDB,
        DumpStats = cms.untracked.bool(True),
        toGet = cms.VPSet(
            cms.PSet(
                record = cms.string('PPSTimingCalibrationRcd'),
                tag = cms.string('PPSDiamondSampicCalibration')
        )
    )
)

process.load("CalibPPS.TimingCalibration.PPSDiamondSampicTimingCalibrationPCLHarvester_cfi")
#process.PPSDiamondSampicTimingCalibrationPCLHarvester.jsonCalibFile="initial.cal.json"
process.PPSDiamondSampicTimingCalibrationPCLHarvester.timingCalibrationTag= ''

# load DQM framework
process.load("DQMServices.Core.DQMStore_cfi")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = "CalibPPS"
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = "/CalibPPS/TimingCalibration/CMSSW_12_6_0_pre2"
process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.forceRunNumber = 999999

process.p = cms.Path(
    process.PPSDiamondSampicTimingCalibrationPCLHarvester
)

process.end_path = cms.EndPath(
    process.dqmEnv +
    process.dqmSaver
)

process.schedule = cms.Schedule(
    process.p,
    process.end_path
)



