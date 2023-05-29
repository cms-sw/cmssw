import FWCore.ParameterSet.Config as cms
import sys

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process('GEMDQM', Run3)

unitTest = False
if 'unitTest=True' in sys.argv:
    unitTest=True

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

if unitTest:
    process.load("DQM.Integration.config.unittestinputsource_cfi")
    from DQM.Integration.config.unittestinputsource_cfi import options
else:
    process.load("DQM.Integration.config.inputsource_cfi")
    from DQM.Integration.config.inputsource_cfi import options

process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "GEM"
process.dqmSaver.tag = "GEM"
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = "GEM"
process.dqmSaverPB.runNumber = options.runNumber

process.load("DQMServices.Components.DQMProvInfo_cfi")

process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('RecoLocalMuon.GEMCSCSegment.gemcscSegments_cff')
process.load("DQM.GEM.GEMDQM_cff")
process.load("DQM.GEM.gemEffByGEMCSCSegment_cff")

process.muonCSCDigis.InputObjects = "rawDataCollector"
if (process.runType.getRunType() == process.runType.hi_run):
    process.muonGEMDigis.InputLabel = "rawDataRepacker"
    process.muonCSCDigis.InputObjects = "rawDataRepacker"

process.muonGEMDigis.useDBEMap = True
process.muonGEMDigis.keepDAQStatus = True

process.gemRecHits.ge21Off = cms.bool(False)

process.GEMDigiSource.runType = "online"
process.GEMRecHitSource.runType = "online"
process.GEMDAQStatusSource.runType = "online"

# from csc_dqm_sourceclient-live_cfg.py
process.CSCGeometryESModule.useGangedStripsInME1a = False
process.idealForDigiCSCGeometry.useGangedStripsInME1a = False
process.CSCIndexerESProducer.AlgoName = "CSCIndexerPostls1"
process.CSCChannelMapperESProducer.AlgoName = "CSCChannelMapperPostls1"
process.csc2DRecHits.readBadChambers = False
process.csc2DRecHits.readBadChannels = False
process.csc2DRecHits.CSCUseGasGainCorrections = False

process.path = cms.Path(
    process.muonGEMDigis *
    process.gemRecHits *
    process.muonCSCDigis *
    process.csc2DRecHits *
    process.cscSegments *
    process.gemcscSegments *
    process.GEMDQM *
    process.gemEffByGEMCSCSegment
)

process.end_path = cms.EndPath(
    process.dqmEnv +
    process.dqmSaver +
    process.dqmSaverPB
)

process.schedule = cms.Schedule(
    process.path,
    process.end_path
)

process.dqmProvInfo.runType = process.runType.getRunTypeName()

from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
print("Final Source settings:", process.source)
