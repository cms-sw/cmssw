run = 357902
input_file=['/store/express/Run2022D/StreamALCAPPSExpress/ALCAPROMPT/PromptCalibProdPPSTimingCalib-Express-v2/000/357/902/00000/123861b4-b632-4835-91b6-1e586d34509e.root']
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
process = cms.Process("harvester", eras.Run3)

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag = GlobalTag(process.GlobalTag, autoCond['run3_data_prompt'], '')

# Source (histograms)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    input_file
    ),
)

# output service for database
process.load('CondCore.CondDB.CondDB_cfi')
process.CondDB.connect = 'sqlite_file:ppsDiamondTiming_calibration'+str(run)+'.sqlite' # SQLite output

process.PoolDBOutputService = cms.Service('PoolDBOutputService',
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('PPSTimingCalibrationRcd_HPTDC'),
            tag = cms.string('DiamondTimingCalibration'),
        )
    )
)

process.load("CalibPPS.TimingCalibration.ppsTimingCalibrationPCLHarvester_cfi")
#process.PPSDiamondSampicTimingCalibrationPCLHarvester.jsonCalibFile="initial.cal.json"

# load DQM framework
process.load("DQMServices.Core.DQMStore_cfi")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = "CalibPPS"
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = "/CalibPPS/TimingCalibration/CMSSW_12_6_0_pre2"
process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.forceRunNumber = run

process.DQMStore = cms.Service("DQMStore")

process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string("harvester_output.root")
)

process.load("Geometry.VeryForwardGeometry.geometryRPFromDB_cfi")
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.EDMtoMEConverter.lumiInputTag = "MEtoEDMConvertPPSTimingCalib:MEtoEDMConverterLumi"
process.EDMtoMEConverter.runInputTag = "MEtoEDMConvertPPSTimingCalib:MEtoEDMConverterRun"

#import FWCore.PythonUtilities.LumiList as LumiList
#process.source.lumisToProcess = LumiList.LumiList(filename = 'allrunsSB-PPS-forCalib.json').getVLuminosityBlockRange() 

process.p = cms.Path(
    process.EDMtoMEConverter*
    process.ppsTimingCalibrationPCLHarvester
)

process.end_path = cms.EndPath(
    process.dqmEnv +
    process.dqmSaver
)

process.schedule = cms.Schedule(
    process.p,
    process.end_path
)



