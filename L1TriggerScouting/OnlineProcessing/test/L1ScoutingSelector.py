import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import sys

options = VarParsing.VarParsing ('analysis')

options.register ("numOrbits",
                  -1,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Number of orbits/events to process")

options.register ("inFile",
                  "file:",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Path to the input file")

options.register ("outFile",
                  "file:/tmp/out.root",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Path of the output file")

options.register ("nThreads",
                  4,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Number of threads")

options.register ("nStreams",
                  4,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Number of streams")

options.register ("debug",
                  False,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.bool,
                  "Run in debug mode")

options.parseArguments()

process = cms.Process( "SCRATES" )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.numOrbits)
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = "WARNING"
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.Timing = cms.Service("Timing",
    summaryOnly = cms.untracked.bool(True),
    useJobReport = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(options.inFile)
)

process.DijetEt30 = cms.EDProducer("JetBxSelector",
    jetsTag          = cms.InputTag("l1ScCaloUnpacker", "Jet"),
    minNJet          = cms.int32(2),
    minJetEt         = cms.vdouble(30, 30),
    maxJetEta        = cms.vdouble(99.9, 99.9)
)

process.HMJetMult4Et20 = cms.EDProducer("JetBxSelector",
    jetsTag          = cms.InputTag("l1ScCaloUnpacker", "Jet"),
    minNJet          = cms.int32(4),
    minJetEt         = cms.vdouble(20, 20, 20, 20),
    maxJetEta        = cms.vdouble(99.9, 99.9, 99.9, 99.9),
)

process.SingleMuPt0BMTF = cms.EDProducer("MuBxSelector",
    muonsTag         = cms.InputTag("l1ScGmtUnpacker",  "Muon"),
    minNMu           = cms.int32(1),
    minMuPt          = cms.vdouble(0.0),
    maxMuEta         = cms.vdouble(99.9),
    minMuTfIndex     = cms.vint32(36),
    maxMuTfIndex     = cms.vint32(71),
    minMuHwQual    = cms.vint32(0),
)

process.DoubleMuPt0Qual8 = cms.EDProducer("MuBxSelector",
    muonsTag         = cms.InputTag("l1ScGmtUnpacker",  "Muon"),
    minNMu           = cms.int32(2),
    minMuPt        = cms.vdouble(0, 0),
    maxMuEta       = cms.vdouble(99.9, 99.9),
    minMuTfIndex     = cms.vint32(0, 0),
    maxMuTfIndex     = cms.vint32(107, 107),
    minMuHwQual      = cms.vint32(8, 8),
)

process.MuTagJetEt30Dr0p4 = cms.EDProducer("MuTagJetBxSelector",
    muonsTag         = cms.InputTag("l1ScGmtUnpacker",  "Muon"),
    jetsTag          = cms.InputTag("l1ScCaloUnpacker", "Jet"),
    minNJet          = cms.int32(1),
    minJetEt         = cms.vdouble(30),
    maxJetEta        = cms.vdouble(99.9),
    minMuPt          = cms.vdouble(0),
    maxMuEta         = cms.vdouble(99.9),
    minMuTfIndex     = cms.vint32(0),
    maxMuTfIndex     = cms.vint32(107),
    minMuHwQual      = cms.vint32(8),
    maxDR            = cms.vdouble(0.4),
)

process.Stubs3BxWindowSimpleCond = cms.EDProducer("BMTFStubMultiBxSelector",
    stubsTag         = cms.InputTag("l1ScBMTFUnpacker", "BMTFStub"),
    condition        = cms.string("simple"),
    bxWindowLength   = cms.uint32(3),
    minNBMTFStub     = cms.uint32(2)
)

process.Stubs3BxWindowWheelCond = cms.EDProducer("BMTFStubMultiBxSelector",
    stubsTag         = cms.InputTag("l1ScBMTFUnpacker", "BMTFStub"),
    condition        = cms.string("wheel"),
    bxWindowLength   = cms.uint32(3),
    minNBMTFStub     = cms.uint32(2)
)

process.FinalBxSelector = cms.EDFilter("FinalBxSelector",
    analysisLabels   = cms.VInputTag(
        cms.InputTag("DijetEt30", "SelBx"),
        cms.InputTag("SingleMuPt0BMTF", "SelBx"),
        cms.InputTag("DoubleMuPt0Qual8", "SelBx"),
        cms.InputTag("HMJetMult4Et20", "SelBx"),
        cms.InputTag("MuTagJetEt30Dr0p4", "SelBx"),
        cms.InputTag("Stubs3BxWindowSimpleCond", "SelBx"),
        cms.InputTag("Stubs3BxWindowWheelCond", "SelBx")
    ),
)

process.bxSelectors = cms.Sequence(
    process.DijetEt30 +
    process.HMJetMult4Et20 +
    process.SingleMuPt0BMTF +
    process.DoubleMuPt0Qual8 +
    process.MuTagJetEt30Dr0p4 +
    process.Stubs3BxWindowSimpleCond +
    process.Stubs3BxWindowWheelCond
)

# Final collection producers
process.FinalBxSelectorMuon = cms.EDProducer("MaskOrbitBxScoutingMuon",
    dataTag = cms.InputTag("l1ScGmtUnpacker",  "Muon"),
    selectBxs = cms.InputTag("FinalBxSelector", "SelBx"),
    productLabel = cms.string("Muon")
)

process.FinalBxSelectorJet = cms.EDProducer("MaskOrbitBxScoutingJet",
    dataTag = cms.InputTag("l1ScCaloUnpacker",  "Jet"),
    selectBxs = cms.InputTag("FinalBxSelector", "SelBx"),
    productLabel = cms.string("Jet")
)

process.FinalBxSelectorEGamma = cms.EDProducer("MaskOrbitBxScoutingEGamma",
    dataTag = cms.InputTag("l1ScCaloUnpacker",  "EGamma"),
    selectBxs = cms.InputTag("FinalBxSelector", "SelBx"),
    productLabel = cms.string("EGamma")
)

process.FinalBxSelectorTau = cms.EDProducer("MaskOrbitBxScoutingTau",
    dataTag = cms.InputTag("l1ScCaloUnpacker",  "Tau"),
    selectBxs = cms.InputTag("FinalBxSelector", "SelBx"),
    productLabel = cms.string("Tau")
)

process.FinalBxSelectorBxSums = cms.EDProducer("MaskOrbitBxScoutingBxSums",
    dataTag = cms.InputTag("l1ScCaloUnpacker",  "EtSum"),
    selectBxs = cms.InputTag("FinalBxSelector", "SelBx"),
    productLabel = cms.string("EtSum")
)

process.FinalBxSelectorBMTFStub = cms.EDProducer("MaskOrbitBxScoutingBMTFStub",
    dataTag = cms.InputTag("l1ScBMTFUnpacker", "BMTFStub"),
    selectBxs = cms.InputTag("FinalBxSelector", "SelBx"),
    productLabel = cms.string("BMTFStub")
)

process.MaskedCollections = cms.Sequence(
    process.FinalBxSelectorMuon +
    process.FinalBxSelectorJet +
    process.FinalBxSelectorEGamma +
    process.FinalBxSelectorTau +
    process.FinalBxSelectorBxSums +
    process.FinalBxSelectorBMTFStub
)

process.pL1ScoutingSelected = cms.Path(process.bxSelectors + process.FinalBxSelector+process.MaskedCollections)

process.hltOutputL1ScoutingSelection = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(options.outFile),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("pL1ScoutingSelected")
    ),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_DijetEt30_*_*',
        'keep *_HMJetMult4Et20_*_*',
        'keep *_SingleMuPt0BMTF_*_*',
        'keep *_DoubleMuPt0Qual8_*_*',
        'keep *_MuTagJetEt30Dr0p4_*_*',
        'keep *_Stubs3BxWindowSimpleCond_*_*',
        'keep *_Stubs3BxWindowWheelCond_*_*',
        'keep *_FinalBxSelector*_*_*',
    )
)

process.options.numberOfThreads = options.nThreads
process.options.numberOfStreams = options.nStreams

#options to override compression algorithm and level for the streamer output
C_LEVEL_UNDEFINED = -1
C_ALGO_UNDEFINED = ""
for moduleName in process.__dict__['_Process__outputmodules']:
    modified_module = getattr(process,moduleName)
    if -1 != C_LEVEL_UNDEFINED:
        modified_module.compression_level=cms.untracked.int32(-1)
    if "" != C_ALGO_UNDEFINED:
        modified_module.compression_algorithm=cms.untracked.string("")

process.options.numberOfThreads = options.nThreads
process.options.numberOfStreams = options.nStreams

process.epL1ScoutingSelection = cms.EndPath(process.hltOutputL1ScoutingSelection)
process.HLTSchedule = cms.Schedule(process.epL1ScoutingSelection)

