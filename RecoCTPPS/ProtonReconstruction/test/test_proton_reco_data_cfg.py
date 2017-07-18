import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("CTPPSTestProtonReconstruction", eras.ctpps_2016)

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('WARNING')
  )
)

# raw data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    "/store/data/Run2016B/DoubleEG/MINIAOD/18Apr2017_ver2-v1/00000/00220DCF-073E-E711-AB1A-0025905C2CBA.root"
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(5000)
)

process.load("RecoCTPPS.ProtonReconstruction.ctppsProtonReconstruction_cfi")

process.ctppsProtonReconstructionValidation = cms.EDAnalyzer("CTPPSProtonReconstructionValidation",
    tagRecoProtons = cms.InputTag("ctppsProtonReconstruction"),
    outputFile = cms.string("validation.root")
)

process.eca = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(
    process.ctppsProtonReconstruction
    #* process.eca
    * process.ctppsProtonReconstructionValidation
)

# output configuration
process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("output.root"),
  outputCommands = cms.untracked.vstring(
    "drop *",
    "keep CTPPSLocalTrackLites_*_*_*",
    "keep recoProtonTracks_*_*_*"
  )
)

process.outpath = cms.EndPath(process.output)
