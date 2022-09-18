import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process('DQM', Run3)

process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(None, 'auto:phase1_2022_cosmics', '')

process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "GEM"
process.dqmEnv.eventInfoFolder = "EventInfo"
process.dqmSaver.path = ""
process.dqmSaver.tag = "GEM"

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('analysis')
options.parseArguments()

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles),
    inputCommands = cms.untracked.vstring(
        'keep *',
    )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(options.maxEvents)
)

process.load("EventFilter.GEMRawToDigi.muonGEMDigis_cfi")
process.load('RecoLocalMuon.GEMRecHit.gemRecHits_cfi')
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('RecoLocalMuon.GEMCSCSegment.gemcscSegments_cff')
process.load("DQM.GEM.gemEffByGEMCSCSegment_cff")

process.muonGEMDigis.useDBEMap = True
process.muonGEMDigis.keepDAQStatus = True  # DEFAULT


process.muonCSCDigis.InputObjects = "rawDataCollector"

#--------------------------------------------------
print("Running with run type = ", process.runType.getRunType())
if (process.runType.getRunType() == process.runType.hi_run):
    process.muonCSCDigis.InputObjects = "rawDataRepacker"


####################################
process.path = cms.Path(
    process.muonGEMDigis *
    process.gemRecHits *
    process.muonCSCDigis *
    process.csc2DRecHits *
    process.cscSegments *
    process.gemcscSegments *
    process.gemEffByGEMCSCSegment
)

process.end_path = cms.EndPath(
    process.dqmEnv +
    process.dqmSaver
)

process.schedule = cms.Schedule(
    process.path,
    process.end_path
)
