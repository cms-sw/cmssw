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
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source(
        "PoolSource",
        fileNames = cms.untracked.vstring([
            "/store/data/Run2015A/MinimumBias/RAW/v1/000/247/215/00000/04D76DE1-3C0C-E511-9821-02163E011DD9.root",
            "/store/data/Run2015A/MinimumBias/RAW/v1/000/247/215/00000/2E9F5DD9-2B0C-E511-A51D-02163E01467B.root",
            "/store/data/Run2015A/MinimumBias/RAW/v1/000/247/215/00000/30634843-2F0C-E511-9AB3-02163E0144C3.root",
            "/store/data/Run2015A/MinimumBias/RAW/v1/000/247/215/00000/4825AFD3-2B0C-E511-93A8-02163E013496.root",
            "/store/data/Run2015A/MinimumBias/RAW/v1/000/247/215/00000/9ED5038A-2E0C-E511-B66B-02163E0144F1.root",
            "/store/data/Run2015A/MinimumBias/RAW/v1/000/247/215/00000/A826FAC5-3C0C-E511-A4CA-02163E013653.root",
            "/store/data/Run2015A/MinimumBias/RAW/v1/000/247/215/00000/AE2CA338-3D0C-E511-9547-02163E0143EB.root",
            "/store/data/Run2015A/MinimumBias/RAW/v1/000/247/215/00000/C81E3707-240C-E511-97CA-02163E01207D.root",
            "/store/data/Run2015A/MinimumBias/RAW/v1/000/247/215/00000/E66FC716-3D0C-E511-8F99-02163E014204.root"
        ]),
        eventsToProcess = cms.untracked.VEventRange('247215:1-247215:MAX')
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

# enable debug message logging for our modules
process.MessageLogger = cms.Service(
    "MessageLogger",
    threshold  = cms.untracked.string('DEBUG'),
    categories = cms.untracked.vstring('L1T'),
#    l1t   = cms.untracked.PSet(
#	threshold  = cms.untracked.string('DEBUG')
#    ),
    debugModules = cms.untracked.vstring('*'),
#        'stage1Raw',
#        'caloStage1Digis'
#    ),
#    cout = cms.untracked.PSet(
#    )
)

# TTree output file
process.load("CommonTools.UtilAlgos.TFileService_cfi")
process.TFileService.fileName = cms.string('l1t.root')


# user stuff

# raw data from MP card
# process.load('EventFilter.L1TRawToDigi.amc13DumpToRaw_cfi')
# process.amc13DumpToRaw.filename = cms.untracked.string("../data/stage1_amc13_example.txt")
# process.amc13DumpToRaw.fedId = cms.untracked.int32(1352)

# dump raw data
process.dumpRaw = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("rawDataCollector"),
    feds = cms.untracked.vint32 ( 1352 ),
    dumpPayload = cms.untracked.bool ( True )
)

# raw to digi
process.load('EventFilter.L1TRawToDigi.caloStage1Digis_cfi')
process.caloStage1Digis.InputLabel = cms.InputTag('rawDataCollector')
process.caloStage1Digis.debug = cms.untracked.bool(True)

# Path and EndPath definitions
process.path = cms.Path(
    # process.amc13DumpToRaw
    # +process.dumpRaw
    process.dumpRaw
    +process.caloStage1Digis
)

process.out = cms.EndPath(
    process.output
)
