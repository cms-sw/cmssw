# hltGetConfiguration /users/jjhollar/ImportAlcaPathsTry2/V4 --globaltag auto:run3_hlt --setup /dev/CMSSW_12_1_0/GRun/V14 --data --unprescale --customise HLTrigger/Configuration/customizeHLTforCMSSW.customiseFor2018Input

# /users/jjhollar/ImportAlcaPathsTry2/V4 (CMSSW_12_1_0)

import FWCore.ParameterSet.Config as cms

process = cms.Process( "HLTX" )
process.load("setup_dev_CMSSW_12_1_0_GRun_V14_cff")

process.HLTConfigVersion = cms.PSet(
  tableName = cms.string('/users/jjhollar/ImportAlcaPathsTry2/V4')
)

process.streams = cms.PSet(  ALCAPPS = cms.vstring( 'AlCaPPS' ) )
process.datasets = cms.PSet(  AlCaPPS = cms.vstring( 'HLT_PPSMaxTracksPerArm1_v1',
  'HLT_PPSMaxTracksPerRP4_v1' ) )

process.hltGetConditions = cms.EDAnalyzer( "EventSetupRecordDataGetter",
    verbose = cms.untracked.bool( False ),
    toGet = cms.VPSet( 
    )
)
process.hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
process.hltPSetMap = cms.EDProducer( "ParameterSetBlobProducer" )
process.hltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
process.hltTriggerType = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 1 )
)
process.hltGtStage2Digis = cms.EDProducer( "L1TRawToDigi",
    FedIds = cms.vint32( 1404 ),
    Setup = cms.string( "stage2::GTSetup" ),
    FWId = cms.uint32( 0 ),
    DmxFWId = cms.uint32( 0 ),
    FWOverride = cms.bool( False ),
    TMTCheck = cms.bool( True ),
    CTP7 = cms.untracked.bool( False ),
    MTF7 = cms.untracked.bool( False ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    lenSlinkHeader = cms.untracked.int32( 8 ),
    lenSlinkTrailer = cms.untracked.int32( 8 ),
    lenAMCHeader = cms.untracked.int32( 8 ),
    lenAMCTrailer = cms.untracked.int32( 0 ),
    lenAMC13Header = cms.untracked.int32( 8 ),
    lenAMC13Trailer = cms.untracked.int32( 8 ),
    debug = cms.untracked.bool( False ),
    MinFeds = cms.uint32( 0 )
)
process.hltGtStage2ObjectMap = cms.EDProducer( "L1TGlobalProducer",
    MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    ExtInputTag = cms.InputTag( "hltGtStage2Digis" ),
    AlgoBlkInputTag = cms.InputTag( "hltGtStage2Digis" ),
    GetPrescaleColumnFromData = cms.bool( False ),
    AlgorithmTriggersUnprescaled = cms.bool( True ),
    RequireMenuToMatchAlgoBlkInput = cms.bool( True ),
    AlgorithmTriggersUnmasked = cms.bool( True ),
    ProduceL1GtDaqRecord = cms.bool( True ),
    ProduceL1GtObjectMapRecord = cms.bool( True ),
    EmulateBxInEvent = cms.int32( 1 ),
    L1DataBxInEvent = cms.int32( 5 ),
    AlternativeNrBxBoardDaq = cms.uint32( 0 ),
    BstLengthBytes = cms.int32( -1 ),
    PrescaleSet = cms.uint32( 1 ),
    Verbosity = cms.untracked.int32( 0 ),
    PrintL1Menu = cms.untracked.bool( False ),
    TriggerMenuLuminosity = cms.string( "startup" ),
    PrescaleCSVFile = cms.string( "prescale_L1TGlobal.csv" )
)
process.hltScalersRawToDigi = cms.EDProducer( "ScalersRawToDigi",
    scalersInputTag = cms.InputTag( "rawDataCollector" )
)
process.hltOnlineBeamSpot = cms.EDProducer( "BeamSpotOnlineProducer",
    changeToCMSCoordinates = cms.bool( False ),
    maxZ = cms.double( 40.0 ),
    setSigmaZ = cms.double( 0.0 ),
    beamMode = cms.untracked.uint32( 11 ),
    src = cms.InputTag( "hltScalersRawToDigi" ),
    gtEvmLabel = cms.InputTag( "" ),
    maxRadius = cms.double( 2.0 ),
    useTransientRecord = cms.bool( False )
)
process.hltL1sZeroBias = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' )
)
process.hltPrePPSMaxTracksPerRP4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltCTPPSPixelDigis = cms.EDProducer( "CTPPSPixelRawToDigi",
    isRun3 = cms.bool( False ),
    includeErrors = cms.bool( True ),
    inputLabel = cms.InputTag( "rawDataCollector" ),
    mappingLabel = cms.string( "RPix" )
)
process.hltCTPPSPixelClusters = cms.EDProducer( "CTPPSPixelClusterProducer",
    RPixVerbosity = cms.untracked.int32( 0 ),
    tag = cms.InputTag( "hltCTPPSPixelDigis" ),
    SeedADCThreshold = cms.int32( 2 ),
    ADCThreshold = cms.int32( 2 ),
    ElectronADCGain = cms.double( 135.0 ),
    VCaltoElectronGain = cms.int32( 50 ),
    VCaltoElectronOffset = cms.int32( -411 ),
    doSingleCalibration = cms.bool( False )
)
process.hltCTPPSPixelRecHits = cms.EDProducer( "CTPPSPixelRecHitProducer",
    RPixVerbosity = cms.untracked.int32( 0 ),
    RPixClusterTag = cms.InputTag( "hltCTPPSPixelClusters" )
)
process.hltCTPPSPixelLocalTracks = cms.EDProducer( "CTPPSPixelLocalTrackProducer",
    tag = cms.InputTag( "hltCTPPSPixelRecHits" ),
    patternFinderAlgorithm = cms.string( "RPixRoadFinder" ),
    trackFinderAlgorithm = cms.string( "RPixPlaneCombinatoryTracking" ),
    trackMinNumberOfPoints = cms.uint32( 3 ),
    verbosity = cms.untracked.int32( 0 ),
    maximumChi2OverNDF = cms.double( 5.0 ),
    maximumXLocalDistanceFromTrack = cms.double( 0.2 ),
    maximumYLocalDistanceFromTrack = cms.double( 0.3 ),
    maxHitPerPlane = cms.int32( 20 ),
    maxHitPerRomanPot = cms.int32( 60 ),
    maxTrackPerRomanPot = cms.int32( 10 ),
    maxTrackPerPattern = cms.int32( 5 ),
    numberOfPlanesPerPot = cms.int32( 6 ),
    roadRadius = cms.double( 1.0 ),
    minRoadSize = cms.int32( 3 ),
    maxRoadSize = cms.int32( 20 )
)
process.hltPPSPrCalFilter = cms.EDFilter( "HLTPPSCalFilter",
    pixelLocalTrackInputTag = cms.InputTag( "hltCTPPSPixelLocalTracks" ),
    minTracks = cms.int32( 1 ),
    maxTracks = cms.int32( 4 ),
    do_express = cms.bool( False ),
    triggerType = cms.int32( 91 )
)
process.hltPPSCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 579, 581, 582, 583, 586, 587, 1462, 1463 )
)
process.hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
process.hltPrePPSMaxTracksPerArm1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPPSExpCalFilter = cms.EDFilter( "HLTPPSCalFilter",
    pixelLocalTrackInputTag = cms.InputTag( "hltCTPPSPixelLocalTracks" ),
    minTracks = cms.int32( 1 ),
    maxTracks = cms.int32( 1 ),
    do_express = cms.bool( True ),
    triggerType = cms.int32( 91 )
)
process.hltFEDSelector = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1023, 1024 )
)
process.hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    throw = cms.bool( False ),
    processName = cms.string( "@" ),
    moduleLabelPatternsToMatch = cms.vstring( 'hlt*' ),
    moduleLabelPatternsToSkip = cms.vstring(  )
)
process.hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
process.hltPreALCAPPSOutput = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)

process.hltOutputALCAPPS = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCAPPS_single.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_PPSMaxTracksPerArm1_v1',
  'HLT_PPSMaxTracksPerRP4_v1' ) ),
    outputCommands = cms.untracked.vstring( 
      'keep *_hltPPSCalibrationRaw_*_*',
      'keep *_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)

process.HLTL1UnpackerSequence = cms.Sequence( process.hltGtStage2Digis + process.hltGtStage2ObjectMap )
process.HLTBeamSpot = cms.Sequence( process.hltScalersRawToDigi + process.hltOnlineBeamSpot )
process.HLTBeginSequence = cms.Sequence( process.hltTriggerType + process.HLTL1UnpackerSequence + process.HLTBeamSpot )
process.HLTPPSPixelRecoSequence = cms.Sequence( process.hltCTPPSPixelDigis + process.hltCTPPSPixelClusters + process.hltCTPPSPixelRecHits + process.hltCTPPSPixelLocalTracks )
process.HLTEndSequence = cms.Sequence( process.hltBoolEnd )

process.HLTriggerFirstPath = cms.Path( process.hltGetConditions + process.hltGetRaw + process.hltPSetMap + process.hltBoolFalse )
process.HLT_PPSMaxTracksPerRP4_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBias + process.hltPrePPSMaxTracksPerRP4 + process.HLTPPSPixelRecoSequence + process.hltPPSPrCalFilter + process.hltPPSCalibrationRaw + process.HLTEndSequence )
process.HLT_PPSMaxTracksPerArm1_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBias + process.hltPrePPSMaxTracksPerArm1 + process.HLTPPSPixelRecoSequence + process.hltPPSExpCalFilter + process.hltPPSCalibrationRaw + process.HLTEndSequence )
process.HLTriggerFinalPath = cms.Path( process.hltGtStage2Digis + process.hltScalersRawToDigi + process.hltFEDSelector + process.hltTriggerSummaryAOD + process.hltTriggerSummaryRAW + process.hltBoolFalse )
process.ALCAPPSOutput = cms.EndPath( process.hltGtStage2Digis + process.hltPreALCAPPSOutput + process.hltOutputALCAPPS )


process.HLTSchedule = cms.Schedule( *(process.HLTriggerFirstPath, process.HLT_PPSMaxTracksPerRP4_v1, process.HLT_PPSMaxTracksPerArm1_v1, process.HLTriggerFinalPath, process.ALCAPPSOutput, ))


process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        'file:RelVal_Raw_GRun_DATA.root',
    ),
    inputCommands = cms.untracked.vstring(
        'keep *'
    )
)

# avoid PrescaleService error due to missing HLT paths
if 'PrescaleService' in process.__dict__:
    for pset in reversed(process.PrescaleService.prescaleTable):
        if not hasattr(process,pset.pathName.value()):
            process.PrescaleService.prescaleTable.remove(pset)

# limit the number of events to be processed
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 100 )
)

# enable TrigReport, TimeReport and MultiThreading
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True ),
    numberOfThreads = cms.untracked.uint32( 4 ),
    numberOfStreams = cms.untracked.uint32( 0 ),
)

# override the GlobalTag, connection string and pfnPrefix
if 'GlobalTag' in process.__dict__:
    from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
    process.GlobalTag = customiseGlobalTag(process.GlobalTag, globaltag = 'auto:run3_hlt')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.TriggerSummaryProducerAOD = cms.untracked.PSet()
    process.MessageLogger.L1GtTrigReport = cms.untracked.PSet()
    process.MessageLogger.L1TGlobalSummary = cms.untracked.PSet()
    process.MessageLogger.HLTrigReport = cms.untracked.PSet()
    process.MessageLogger.FastReport = cms.untracked.PSet()
    process.MessageLogger.ThroughputService = cms.untracked.PSet()

# load the DQMStore and DQMRootOutputModule
process.load( "DQMServices.Core.DQMStore_cfi" )

process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string("DQMIO.root")
)

process.DQMOutput = cms.EndPath( process.dqmOutput )

# add specific customizations
_customInfo = {}
_customInfo['menuType'  ]= "GRun"
_customInfo['globalTags']= {}
_customInfo['globalTags'][True ] = "auto:run3_hlt_GRun"
_customInfo['globalTags'][False] = "auto:run3_mc_GRun"
_customInfo['inputFiles']={}
_customInfo['inputFiles'][True]  = "file:RelVal_Raw_GRun_DATA.root"
_customInfo['inputFiles'][False] = "file:RelVal_Raw_GRun_MC.root"
_customInfo['maxEvents' ]=  -1
_customInfo['globalTag' ]= "auto:run3_hlt"
_customInfo['inputFile' ]=  ['file:/eos/user/m/maaraujo/HLT_work/HLT_rem_322022/Skim_Run322022_LS1025to1344_1.root']
# _customInfo['inputFile' ]=  [' /store/data/Run2018D/ZeroBias/RAW/v1/000/322/022/00000/7E894F17-9AAD-E811-9112-FA163E147026.root']
_customInfo['realData'  ]=  True
from HLTrigger.Configuration.customizeHLTforALL import customizeHLTforAll
process = customizeHLTforAll(process,"GRun",_customInfo)

from HLTrigger.Configuration.customizeHLTforCMSSW import customizeHLTforCMSSW
process = customizeHLTforCMSSW(process,"GRun")

# Eras-based customisations
from HLTrigger.Configuration.Eras import modifyHLTforEras
modifyHLTforEras(process)

#User-defined customization functions
from HLTrigger.Configuration.customizeHLTforCMSSW import customiseFor2018Input
process = customiseFor2018Input(process)

