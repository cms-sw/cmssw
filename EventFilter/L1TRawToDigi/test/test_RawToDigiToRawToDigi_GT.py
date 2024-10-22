import FWCore.ParameterSet.Config as cms

process = cms.Process('L1')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source(
        "PoolSource",
        fileNames = cms.untracked.vstring([
            "/store/hidata/HIRun2023A/HIMinimumBias1/RAW/v1/000/374/345/00000/1332e7ef-bf52-4bc4-b6e2-aae1be281411.root",
            "/store/hidata/HIRun2023A/HIMinimumBias1/RAW/v1/000/374/345/00000/6b72b001-db9b-4490-aec8-29eb1de07108.root",
            "/store/hidata/HIRun2023A/HIMinimumBias1/RAW/v1/000/374/345/00000/1859b208-0daf-4382-a2fc-444a8e7628aa.root",
            "/store/hidata/HIRun2023A/HIMinimumBias1/RAW/v1/000/374/345/00000/77101f68-bc1b-4ca6-a8ba-f48bef24b8de.root",
            "/store/hidata/HIRun2023A/HIMinimumBias1/RAW/v1/000/374/345/00000/1fc90d44-5798-4eb1-a3c0-2c2e2e9a9df6.root"
        ]),
        eventsToProcess = cms.untracked.VEventRange('374345:1-374345:MAX')
)

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)


# Output definition

process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring("keep *",
					   "drop *_mix_*_*"),
    fileName = cms.untracked.string('L1T_EDM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')


# # TTree output file
# process.load("CommonTools.UtilAlgos.TFileService_cfi")
# process.TFileService.fileName = cms.string('l1t.root')


process.load('EventFilter.L1TRawToDigi.gtStage2Digis_cfi')
process.gtStage2Digis.InputLabel = cms.InputTag('rawDataRepacker')
process.gtStage2Digis.debug = cms.untracked.bool(False)

process.load('EventFilter.L1TRawToDigi.gtStage2Raw_cfi')
process.gtStage2Raw.InputLabel = cms.InputTag("gtStage2Digis","GT")

process.gtStage2Raw.GtInputTag = cms.InputTag("gtStage2Digis","")
process.gtStage2Raw.ExtInputTag = cms.InputTag("gtStage2Digis")
process.gtStage2Raw.MuonInputTag   = cms.InputTag("gtStage2Digis","Muon")
process.gtStage2Raw.EGammaInputTag = cms.InputTag("gtStage2Digis","EGamma")
process.gtStage2Raw.TauInputTag    = cms.InputTag("gtStage2Digis","Tau")
process.gtStage2Raw.JetInputTag    = cms.InputTag("gtStage2Digis","Jet")
process.gtStage2Raw.EtSumInputTag  = cms.InputTag("gtStage2Digis","EtSum")  
process.gtStage2Raw.EtSumZDCInputTag  = cms.InputTag("gtStage2Digis","EtSumZDC")  


# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    token = cms.untracked.InputTag('gtStage2Raw'),
    label = cms.untracked.InputTag('gtStage2Raw'),
    feds = cms.untracked.vint32 ( 1404 ),
    dumpPayload = cms.untracked.bool ( False )
)
process.unpackGtStage2 = process.gtStage2Digis.clone()
process.unpackGtStage2.InputLabel = cms.InputTag('gtStage2Raw')
process.unpackGtStage2.debug = cms.untracked.bool(False)


# Path and EndPath definitions
process.path = cms.Path(
    process.gtStage2Digis
    + process.gtStage2Raw
    # + process.dumpRaw
    + process.unpackGtStage2
)

process.out = cms.EndPath(
    process.output
)
