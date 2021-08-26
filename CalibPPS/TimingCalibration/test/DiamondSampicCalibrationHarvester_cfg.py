import FWCore.ParameterSet.Config as cms

process = cms.Process("harvester")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '113X_dataRun3_Prompt_Candidate_2021_08_24_17_11_47')

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
            record = cms.string('PPSTimingCalibrationRcd'),
            tag = cms.string('DiamondSampicCalibration'),
        )
    )
)

################
#geometry
################
process.load('Geometry.VeryForwardGeometry.geometryRPFromDD_2021_cfi')

process.load("CalibPPS.TimingCalibration.PPSDiamondSampicTimingCalibrationPCLHarvester_cfi")
#process.PPSDiamondSampicTimingCalibrationPCLHarvester.jsonCalibFile=cms.string("initial.cal.json")

process.p = cms.Path(
    process.PPSDiamondSampicTimingCalibrationPCLHarvester
)

process.schedule = cms.Schedule(
    process.p
)


