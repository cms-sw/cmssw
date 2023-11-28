import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('RECODQM', Run3)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.verbosity = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    )
)

# import of standard configurations
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


# raw data source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Run2023D/ZeroBias/RAW/v1/000/369/956/00000/33d5acec-484f-4ac0-9b83-c6a3104ddd2b.root'
    ),
)


from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag.globaltag = "130X_dataRun3_Prompt_forLHCInfo_Candidate_2023_08_08_10_52_01"

# local RP reconstruction chain with standard settings
process.load("RecoPPS.Configuration.recoCTPPS_cff")


process.ctppsProtonReconstructionPlotter = cms.EDAnalyzer("CTPPSProtonReconstructionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"),
    tagRecoProtonsSingleRP = cms.InputTag("ctppsProtons", "singleRP"),
    tagRecoProtonsMultiRP = cms.InputTag("ctppsProtons", "multiRP"),

    rpId_45_F = cms.uint32(23),
    rpId_45_N = cms.uint32(3),
    rpId_56_N = cms.uint32(103),
    rpId_56_F = cms.uint32(123),

    outputFile = cms.string("alcareco_protons_express.root"),
)


process.path = cms.Path(
    process.ctppsRawToDigi
    * process.recoCTPPS
)

process.end_path = cms.EndPath(
    process.ctppsProtonReconstructionPlotter
)

process.schedule = cms.Schedule(
    process.path,
    process.end_path
)
