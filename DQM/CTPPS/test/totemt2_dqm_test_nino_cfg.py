import FWCore.ParameterSet.Config as cms
import string

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('RECODQM', Run3)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(6000) )
process.verbosity = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
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
# raw data source
process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(
#    '/store/t0streamer/Minidaq/A/000/368/023/run368023_ls0001_streamA_StorageManager.dat',
    'file:/eos/cms/store/t0streamer/Minidaq/A/000/368/080/run368080_ls0001_streamA_StorageManager.dat',
    'file:/eos/cms/store/t0streamer/Minidaq/A/000/368/080/run368080_ls0002_streamA_StorageManager.dat',
#    '/store/t0streamer/Minidaq/A/000/368/081/run368081_ls0001_streamA_StorageManager.dat',
#    '/store/t0streamer/Minidaq/A/000/368/081/run368081_ls0002_streamA_StorageManager.dat',
#    '/store/t0streamer/Minidaq/A/000/368/082/run368082_ls0001_streamA_StorageManager.dat',
#    '/store/t0streamer/Minidaq/A/000/368/082/run368082_ls0002_streamA_StorageManager.dat',
#        '/store/group/dpg_ctpps/comm_ctpps/TotemT2/RecoTest/run364983_ls0001_streamA_StorageManager.dat',
    )
)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '130X_dataRun3_HLT_v2', '')

#Raw-to-digi
process.load('EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff')

# local RP reconstruction chain with standard settings
process.load("RecoPPS.Configuration.recoCTPPS_cff")

# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")
process.totemDAQMappingESSourceXML_TotemT2.verbosity = 0
process.totemT2Digis.RawUnpacking.verbosity = 0
process.totemT2Digis.RawToDigi.verbosity = 0
process.totemT2Digis.RawToDigi.testCRC = 1
process.totemT2Digis.RawToDigi.useOlderT2TestFile = False
process.totemT2Digis.RawToDigi.printUnknownFrameSummary = True
process.totemT2Digis.RawToDigi.printErrorSummary = True

process.path = cms.Path(
    process.ctppsRawToDigi *
    process.totemT2Digis *
    process.totemT2RecHits *
    process.totemT2DQMSource
)

process.end_path = cms.EndPath(
    process.dqmEnv +
    process.dqmSaver
)

process.schedule = cms.Schedule(
    process.path,
    process.end_path
)
