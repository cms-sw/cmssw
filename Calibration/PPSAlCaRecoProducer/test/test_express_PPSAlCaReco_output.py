import FWCore.ParameterSet.Config as cms

# load era modifier to run on 2022 data
from Configuration.Eras.Modifier_ctpps_2022_cff import ctpps_2022

process = cms.Process( 'TEST',ctpps_2022)

# command  line options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing('analysis')
options.register('runNo',
                1,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "Run number")
options.parseArguments()

run_no = options.runNo

# LHCInfo plotter
process.load('Validation.CTPPS.ctppsLHCInfoPlotter_cfi')
process.ctppsLHCInfoPlotter.outputFile = f"alcareco_lhc_info_express_{run_no}.root"

# Load geometry from DB
process.load("Geometry.VeryForwardGeometry.geometryRPFromDB_cfi")

# track distribution plotter
process.ctppsTrackDistributionPlotter = cms.EDAnalyzer("CTPPSTrackDistributionPlotter",
  tagTracks = cms.InputTag("ctppsLocalTrackLiteProducerAlCaRecoProducer"),

  rpId_45_F = cms.uint32(23),
  rpId_45_N = cms.uint32(3),
  rpId_56_N = cms.uint32(103),
  rpId_56_F = cms.uint32(123),

  outputFile = cms.string(f"alcareco_tracks_express_{run_no}.root")
)

# reconstruction plotter
process.ctppsProtonReconstructionPlotter = cms.EDAnalyzer("CTPPSProtonReconstructionPlotter",
  tagTracks = cms.InputTag("ctppsLocalTrackLiteProducerAlCaRecoProducer"),
  tagRecoProtonsSingleRP = cms.InputTag("ctppsProtonsAlCaRecoProducer", "singleRP"),
  tagRecoProtonsMultiRP = cms.InputTag("ctppsProtonsAlCaRecoProducer", "multiRP"),

  rpId_45_F = cms.uint32(23),
  rpId_45_N = cms.uint32(3),
  rpId_56_N = cms.uint32(103),
  rpId_56_F = cms.uint32(123),

  outputFile = cms.string(f"alcareco_protons_express_{run_no}.root")
)

process.p = cms.Path(
  process.ctppsLHCInfoPlotter
  * process.ctppsTrackDistributionPlotter
  * process.ctppsProtonReconstructionPlotter
)

process.options = cms.PSet(
    wantSummary = cms.untracked.bool( True ),
)

# load GT
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data_express')

process.source = cms.Source( 'PoolSource',
    fileNames = cms.untracked.vstring(
        options.inputFiles,
    ),
    inputCommands = cms.untracked.vstring(
        'keep *'
    )
)

# limit the number of events to be processed
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)
