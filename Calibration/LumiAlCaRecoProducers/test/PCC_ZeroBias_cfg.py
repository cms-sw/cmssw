#########################
#Author: Sam Higginbotham
#Purpose: To investigate the AlCaPCCProducer input and output. 
#########################
import FWCore.ParameterSet.Config as cms

process = cms.Process("ALCARECO")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/eos/cms/store/data/Run2015D/AlCaLumiPixels/ALCARECO/LumiPixels-PromptReco-v4/000/260/039/00000/1CF2A210-5B7E-E511-8F4F-02163E014145.root')
)

#Added process to select the appropriate events 
process.OutALCARECOPromptCalibProdPCC = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdPCC')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_alcaPCCProducer_*_*', 
        'keep *_MEtoEDMConvertSiStrip_*_*')
)
#Make sure that variables match in producer.cc and .h
process.alcaPCCProducer = cms.EDProducer("AlcaPCCProducer",
    AlcaPCCProducerParameters = cms.PSet(
        WriteToDB = cms.bool(False),
        pixelClusterLabel = cms.InputTag("siPixelClustersForLumi"), 
        #Mod factor to count lumi and the string to specify output 
        resetEveryNLumi = cms.untracked.int32(1),
        trigstring = cms.untracked.string("alcaPCCZB") 
    ),
)

process.OutALCARECOLumiPixels = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOLumiPixels')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_siPixelClustersForLumi_*_*', 
        'keep *_TriggerResults_*_HLT')
)


process.OutALCARECOLumiPixels_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOLumiPixels')
    ),
    outputCommands = cms.untracked.vstring('keep *_siPixelClustersForLumi_*_*', 
        'keep *_TriggerResults_*_HLT')
)

process.siPixelClustersForLumi = cms.EDProducer("SiPixelClusterProducer",
    ChannelThreshold = cms.int32(1000),
    ClusterThreshold = cms.double(4000.0),
    MissCalibrate = cms.untracked.bool(True),
    SeedThreshold = cms.int32(1000),
    SplitClusters = cms.bool(False),
    VCaltoElectronGain = cms.int32(65),
    VCaltoElectronOffset = cms.int32(-414),
    maxNumberOfClusters = cms.int32(-1),
    payloadType = cms.string('Offline'),
    src = cms.InputTag("siPixelDigisForLumi")
)


process.siPixelDigisForLumi = cms.EDProducer("SiPixelRawToDigi",
    CablingMapLabel = cms.string(''),
    ErrorList = cms.vint32(29),
    IncludeErrors = cms.bool(True),
    InputLabel = cms.InputTag("hltFEDSelectorLumiPixels"),
    Regions = cms.PSet(

    ),
    Timing = cms.untracked.bool(False),
    UsePhase1 = cms.bool(False),
    UsePilotBlade = cms.bool(False),
    UseQualityInfo = cms.bool(False),
    UserErrorList = cms.vint32(40)
)




#HLT filter for PCC
process.ALCARECOHltFilterForPCC = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring("*ZeroBias*"),
    eventSetupPathsKey = cms.string(""),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
    andOr = cms.bool(True),
    throw = cms.bool(False)
)
#From the end path, this is where we specify format for our output.
process.ALCARECOStreamPromptCalibProdPCC = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdPCC')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('ALCAPROMPT'),
        filterName = cms.untracked.string('PromptCalibProdPCC')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('ProdPCC_ZeroBias_1.root'),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_alcaPCCProducer_*_*', 
        'keep *_MEtoEDMConvertSiStrip_*_*')
)


#
process.alcaPCC = cms.Sequence(process.alcaPCCProducer)

#This is the key sequence that we are adding first...
process.seqALCARECOPromptCalibProdPCC = cms.Sequence(process.ALCARECOHltFilterForPCC+process.alcaPCCProducer)

process.pathALCARECOPromptCalibProdPCC = cms.Path(process.seqALCARECOPromptCalibProdPCC)

process.seqALCARECOLumiPixels = cms.Sequence(process.siPixelDigisForLumi+process.siPixelClustersForLumi)

process.pathALCARECOLumiPixels = cms.Path(process.seqALCARECOLumiPixels)

process.ALCARECOStreamPromptCalibProdOutPath = cms.EndPath(process.ALCARECOStreamPromptCalibProdPCC)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkReport = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
            reportEvery = cms.untracked.int32(10000)
        ),
        FwkSummary = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
            reportEvery = cms.untracked.int32(1)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            limit = cms.untracked.int32(0),
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noTimeStamps = cms.untracked.bool(False),
        threshold = cms.untracked.string('INFO'),
        enableStatistics = cms.untracked.bool(True),
        statisticsThreshold = cms.untracked.string('WARNING')
    ),
    debugModules = cms.untracked.vstring(),
    default = cms.untracked.PSet(

    ),
    suppressDebug = cms.untracked.vstring(),
    suppressInfo = cms.untracked.vstring(),
    suppressWarning = cms.untracked.vstring()
)
#added line for additional output summary `
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )


process.schedule = cms.Schedule(*[ process.pathALCARECOPromptCalibProdPCC, process.ALCARECOStreamPromptCalibProdOutPath ])
