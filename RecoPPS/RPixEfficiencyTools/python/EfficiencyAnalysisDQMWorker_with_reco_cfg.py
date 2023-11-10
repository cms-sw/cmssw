import os
import sys

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from Configuration.StandardSequences.Eras import eras
from Configuration.AlCa.GlobalTag import GlobalTag
import FWCore.ParameterSet.VarParsing as VarParsing


# GLOBAL CONSTANT VARIABLES
# fiducial variables restrict the area to analyze 
# the current parameters cover the whole possible area
fiducialXLow = [0,0,0,0]
fiducialYLow = [-99.,-99.,-99.,-99.]
fiducialYHigh = [99.,99.,99.,99.]

#SETUP PROCESS
process = cms.Process("EfficiencyAnalysisDQMWorker", eras.Run3)


#SETUP PARAMETERS
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    FailPath = cms.untracked.vstring('Type Mismatch') # not crashing on this exception type
    )
options = VarParsing.VarParsing('analysis')

options.register('alignmentXMLName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "Alignment XML file name")
options.register('alignmentDBName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "Alignmend DB file name")

options.register('outputFileName',
                'outputEfficiencyAnalysisDQMWorker.root',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "output ROOT file name")
options.register('sourceFileList',
                '../test/testData_rereco+eff.dat',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "source file list name")
options.register('runNumber',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "CMS Run Number")
options.register('bunchSelection',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "bunches to be analyzed")
options.register('jsonFileName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "JSON file list name")
options.register('injectionSchemeFileName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.bool,
                "Injection scheme file name")
options.register('supplementaryPlots',
                True,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.bool,
                "should add bin shifted hitTrackDistribution")
options.register('globalTag',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "GT to use")


#INTERPOT
options.register('maxTracksInTagPot',
                99,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "Maximum pixel tracks in tag RP")
options.register('minTracksInTagPot',
                0,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "Minimum pixel tracks in tag RP")
options.register('maxTracksInProbePot',
                99,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "Maximum pixel tracks in probe RP")
options.register('minTracksInProbePot',
                0,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "Mainimum pixel tracks in probe RP")
options.register('maxChi2Prob',
                0.999999,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.float,
                "Maximum chi2 probability of the track")
options.register('recoInfo',
                -1,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "CTPPSpixelLocalTrackReconstructionInfo proton variable - -1 for no selection")

options.parseArguments()


#PROCESS PARAMETERS

# Prefer input from files over source
if len(options.inputFiles) != 0:
    fileList = [f'file:{f}' if not (f.startswith('/store/') or f.startswith('file:') or f.startswith('root:')) else f for f in options.inputFiles]
    inputFiles = cms.untracked.vstring(fileList)
    print('Input files:')
    print(inputFiles)
elif options.sourceFileList != '':
    import FWCore.Utilities.FileUtils as FileUtils
    print('Taking input from:',options.sourceFileList)
    fileList = FileUtils.loadListFromFile (options.sourceFileList) 
    inputFiles = cms.untracked.vstring( *fileList)

# runToScheme = {}
# with open("./data/RunToScheme2018.csv") as runToSchemeFile:
#     firstcycle = True
#     next(runToSchemeFile)
#     for line in runToSchemeFile:
#        (run, fill, injectionScheme) = line.split(", ")
#        runToScheme[int(run)] = injectionScheme.rstrip()

# if options.bunchSelection != 'NoSelection' and options.bunchSelection != '':
#     if options.runNumber in runToScheme.keys():
#         injectionSchemeFileName = './data/2018_FillingSchemes/'+runToScheme[options.runNumber]+'.csv'
#     else:
#         injectionSchemeFileName = options.injectionSchemeFileName
#     print("Using filling scheme: "+injectionSchemeFileName)
# else:
#     injectionSchemeFileName = ''
injectionSchemeFileName = ''


#LOAD NECCESSARY DEPENDENCIES
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQM.Integration.config.environment_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Geometry.VeryForwardGeometry.geometryRPFromDB_cfi")

#SETUP LOGGER
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout'),
    cout = cms.untracked.PSet( 
        optionalPSet = cms.untracked.bool(True),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            reportEvery = cms.untracked.int32(10000),
            limit = cms.untracked.int32(50000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('INFO')
    ),
    categories = cms.untracked.vstring(
        "FwkReport"
    ),
)

#CONFIGURE PROCESS
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#SETUP GLOBAL TAG
if options.globalTag != '':
    gt = options.globalTag
else:
    gt = 'auto:run3_data_prompt'

print('Using GT:',gt)
process.GlobalTag = GlobalTag(process.GlobalTag, gt)

#SETUP INPUT
process.source = cms.Source("PoolSource",
    fileNames = inputFiles, 
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_*_*_RECO',
        'keep *_*Digi*_*_RECO',
        'keep *_hltGtStage2Digis_*_*',
        'keep *_gtStage2Digis_*_*',
    )
)

if options.jsonFileName:
    print("Using JSON file...")
    import FWCore.PythonUtilities.LumiList as LumiList
    if options.jsonFileName == '':
        jsonFileName = 'test/JSONFiles/Run'+str(options.runNumber)+'.json'
    else:
        jsonFileName = options.jsonFileName
    print(jsonFileName)
    process.source.lumisToProcess = LumiList.LumiList(filename = jsonFileName).getVLuminosityBlockRange()

# Handle alignment inputs
if options.alignmentXMLName and options.alignmentDBName:
    print('ERROR: Both alignment XML and DB files specified. Please specify only one.')
    sys.exit(1)

if options.alignmentXMLName:
    # Load alignments from XML  
    print('Loading alignment from XML file:', options.alignmentXMLName)
    process.load("CalibPPS.ESProducers.ctppsRPAlignmentCorrectionsDataESSourceXML_cfi")
    process.ctppsRPAlignmentCorrectionsDataESSourceXML.RealFiles = cms.vstring(options.alignmentXMLName)
    process.esPreferLocalAlignment = cms.ESPrefer("CTPPSRPAlignmentCorrectionsDataESSourceXML", "ctppsRPAlignmentCorrectionsDataESSourceXML")
elif options.alignmentDBName:
    # Load alignments from DB file
    print('Loading alignment from DB file:', options.alignmentDBName)
    from CondCore.CondDB.CondDB_cfi import *
    ppsAlignmentDB = CondDB.clone()
    ppsAlignmentDB.connect = cms.string("sqlite_file:"+options.alignmentDBName)
    process.ppsAlignment = cms.ESSource("PoolDBESSource",ppsAlignmentDB,
        toGet = cms.VPSet(
            cms.PSet(
            record = cms.string("RPRealAlignmentRecord"),
            tag = cms.string("CTPPSRPAlignment_real"),
            label = cms.untracked.string("")
            )
        )
    )
    process.es_prefer_ppsAlignment = cms.ESPrefer("PoolDBESSource","ppsAlignment")
else: 
    print('Using alignment from GT.')

#SETUP TRACK AND PROTON TAGS TO RUN ON ALCARECO
trackTag = ('ctppsPixelLocalTracksAlCaRecoProducer','','EfficiencyAnalysisDQMWorker')
protonTag = 'ctppsProtonsAlCaRecoProducer'
print('Using track InputTag:',trackTag)
print('Using proton InputTag:',protonTag)

#SETUP WORKER
process.worker = DQMEDAnalyzer('EfficiencyTool_2018DQMWorker',
    tagPixelLocalTracks=cms.untracked.InputTag(trackTag),
    minNumberOfPlanesForEfficiency=cms.int32(3),
    minNumberOfPlanesForTrack=cms.int32(3),
    maxNumberOfPlanesForTrack=cms.int32(6),
    isCorrelationPlotEnabled=cms.bool(False),                       #Only enable if the estimation of the correlation between Strips and Pixel tracks is under study 
                                                                    #(disables filling of TGraph, reducing the output file size)
    minTracksPerEvent=cms.int32(0),
    maxTracksPerEvent=cms.int32(99),
    supplementaryPlots=cms.bool(options.supplementaryPlots),
    bunchSelection=cms.untracked.string(options.bunchSelection),
    bunchListFileName=cms.untracked.string(injectionSchemeFileName),
    binGroupingX=cms.untracked.int32(1),
    binGroupingY=cms.untracked.int32(1),
    fiducialXLow=cms.untracked.vdouble(fiducialXLow),
    fiducialYLow=cms.untracked.vdouble(fiducialYLow),
    fiducialYHigh=cms.untracked.vdouble(fiducialYHigh),
    detectorTiltAngle=cms.untracked.double(20),
    detectorRotationAngle=cms.untracked.double(-8),

    #FOR INTERPORT EFFICIENCY
    tagProtonsSingleRP=cms.untracked.InputTag(protonTag, "singleRP", "EfficiencyAnalysisDQMWorker"),
    tagProtonsMultiRP=cms.untracked.InputTag(protonTag, "multiRP", "EfficiencyAnalysisDQMWorker"),
    maxChi2Prob=cms.untracked.double(options.maxChi2Prob),
    maxTracksInProbePot=cms.untracked.int32(options.maxTracksInProbePot),    
    minTracksInProbePot=cms.untracked.int32(options.minTracksInProbePot),    
    maxTracksInTagPot=cms.untracked.int32(options.maxTracksInTagPot),    
    minTracksInTagPot=cms.untracked.int32(options.minTracksInTagPot),  
    recoInfo=cms.untracked.int32(options.recoInfo),
    debug=cms.untracked.bool(True),
    # Configs for prescale provider
    processName = cms.string("HLT"),
    triggerPattern = cms.string("HLT_PPSMaxTracksPerRP4_v*"),
    stageL1Trigger = cms.uint32(2),
    # l1tAlgBlkInputTag = cms.InputTag('hltGtStage2ObjectMap'),
    # l1tExtBlkInputTag = cms.InputTag('hltGtStage2ObjectMap')
    l1tAlgBlkInputTag = cms.InputTag('gtStage2Digis'),
    l1tExtBlkInputTag = cms.InputTag('gtStage2Digis')
)

#SETUP OUTPUT
print('Output will be saved in',options.outputFileName)
process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string(options.outputFileName),
    outputCommands = cms.untracked.vstring(
        "drop *_*_*_RECO"
    )
)

# Load the ALCARECO reco step from DIGI
process.load("Calibration.PPSAlCaRecoProducer.ALCARECOPPSCalMaxTracks_cff")
# Remove diamond reco
process.recoPPSSequenceAlCaRecoProducer.remove(process.diamondSampicLocalReconstructionTaskAlCaRecoProducer)

#ENABLE IF RUNNING ON NON-ALCARECO DATASET
process.ctppsPixelClustersAlCaRecoProducer.tag='ctppsPixelDigis'
process.ctppsDiamondRecHitsAlCaRecoProducer.digiTag='ctppsDiamondRawToDigi:TimingDiamond'

# Filter only ZB events to be sure that we're getting the right prescale
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter
process.filterL1ZeroBias = triggerResultsFilter.clone(
    hltResults = cms.InputTag('TriggerResults', '', 'HLT'),
    # l1tResults = cms.InputTag("hltGtStage2ObjectMap", "", "HLT"),  # Adjust the InputTag accordingly
    l1tResults = cms.InputTag("gtStage2Digis"),  # Adjust the InputTag accordingly
    triggerConditions = cms.vstring("L1_ZeroBias"),  # Replace with the name of your L1 trigger
)

#SCHEDULE JOB
process.path = cms.Path(
    # process.filterL1ZeroBias *
    process.recoPPSSequenceAlCaRecoProducer *
    process.worker
)

process.end_path = cms.EndPath(
    process.dqmOutput
)

process.schedule = cms.Schedule(
    process.path,
    process.end_path
)
