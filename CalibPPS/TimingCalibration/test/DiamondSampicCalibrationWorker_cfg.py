import FWCore.ParameterSet.Config as cms

process = cms.Process("worker")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('RecoPPS.Local.totemTimingLocalReconstruction_cff')
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag = GlobalTag(process.GlobalTag, autoCond['run3_data_prompt'], '') 

process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(
    '/store/data/Run2022B/AlCaPPS/RAW/v1/000/355/207/00000/c23440f4-49c0-44aa-b8f6-f40598fb4705.root',
    

),
)

process.load('CondCore.CondDB.CondDB_cfi')
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

process.totemTimingRecHits.mergeTimePeaks= False
process.totemTimingRecHits.timingCalibrationTag= ''

process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")
process.totemTimingRawToDigi.rawDataTag = "hltPPSCalibrationRaw"
################
#geometry
################
process.load("Geometry.VeryForwardGeometry.geometryRPFromDB_cfi")

process.load("CalibPPS.TimingCalibration.PPSDiamondSampicTimingCalibrationPCLWorker_cfi")
process.DQMStore = cms.Service("DQMStore")

process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string("worker_output.root")
)


process.path = cms.Path(
    process.totemTimingRawToDigi*
    process.diamondSampicLocalReconstruction*
    process.PPSDiamondSampicTimingCalibrationPCLWorker
)

process.end_path = cms.EndPath(
    process.dqmOutput
)

process.schedule = cms.Schedule(
    process.path,
    process.end_path
)
