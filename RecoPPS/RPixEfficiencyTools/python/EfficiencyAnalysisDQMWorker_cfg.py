import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from Configuration.StandardSequences.Eras import eras
from Configuration.AlCa.GlobalTag import GlobalTag
import FWCore.ParameterSet.VarParsing as VarParsing


#GLOBAL CONSTANT VARIABLES
# fiducial variables restrict the area to analyze 
# - the current parameters cover the whole possible area
fiducialXLow = [0,0,0,0]
fiducialYLow = [-99.,-99.,-99.,-99.]
fiducialYHigh = [99.,99.,99.,99.]

#SETUP PROCESS
process = cms.Process("DQMWorkerProcess", eras.Run2_2018,eras.run2_miniAOD_devel)


#SETUP PARAMETERS
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False),
    FailPath = cms.untracked.vstring('Type Mismatch') # not crashing on this exception type
    )
options = VarParsing.VarParsing ()
options.register('outputFileName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "output ROOT file name")
options.register('sourceFileList',
                '',
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
options.register('useJsonFile',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.bool,
                "Do not use JSON file")
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
options.register('suplementaryPlots',
                False,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.bool,
                "should add bin shifted hitTrackDistribution")


#INTERPOT
options.register('maxTracksInTagPot',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "Maximum pixel tracks in tag RP")
options.register('minTracksInTagPot',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "Minimum pixel tracks in tag RP")
options.register('maxTracksInProbePot',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "Maximum pixel tracks in probe RP")
options.register('minTracksInProbePot',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "Mainimum pixel tracks in probe RP")
options.register('maxChi2Prob',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.float,
                "Maximum chi2 probability of the track")
options.register('recoInfo',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "CTPPSpixelLocalTrackReconstructionInfo proton variable - -1 for no selection")
options.maxChi2Prob = 0.999999
options.maxTracksInTagPot = 99
options.minTracksInTagPot = 0
options.maxTracksInProbePot = 99
options.minTracksInProbePot = 0
options.recoInfo = -1



options.parseArguments()


#PROCESS PARAMETERS
if options.sourceFileList != '':
    import FWCore.Utilities.FileUtils as FileUtils
    fileList = FileUtils.loadListFromFile (options.sourceFileList) 
    inputFiles = cms.untracked.vstring( *fileList)

if options.useJsonFile == True:
    print("Using JSON file...")
    import FWCore.PythonUtilities.LumiList as LumiList
    if options.jsonFileName == '':
        jsonFileName = 'test/JSONFiles/Run'+str(options.runNumber)+'.json'
    else:
        jsonFileName = options.jsonFileName
    print(jsonFileName)
    process.source.lumisToProcess = LumiList.LumiList(filename = jsonFileName).getVLuminosityBlockRange()

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
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("DQM.Integration.config.environment_cfi")
process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi")

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
process.MessageLogger.statistics = cms.untracked.vstring()


#CONFIGURE PROCESS
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )


#SETUP GLOBAL TAG
process.GlobalTag = GlobalTag(process.GlobalTag, '123X_dataRun2_v4')


#SETUP INPUT
process.source = cms.Source("PoolSource",
    fileNames = inputFiles
)


#SETUP WORKER
process.worker = DQMEDAnalyzer('EfficiencyTool_2018DQMWorker',
    minNumberOfPlanesForEfficiency=cms.int32(3),
    minNumberOfPlanesForTrack=cms.int32(3),
    maxNumberOfPlanesForTrack=cms.int32(6),
    isCorrelationPlotEnabled=cms.bool(False),                       #Only enable if the estimation of the correlation between Strips and Pixel tracks is under study 
                                                                    #(disables filling of TGraph, reducing the output file size)
    minTracksPerEvent=cms.int32(0),
    maxTracksPerEvent=cms.int32(99),
    supplementaryPlots=cms.bool(options.suplementaryPlots),
    bunchSelection=cms.untracked.string(options.bunchSelection),
    bunchListFileName=cms.untracked.string(injectionSchemeFileName),
    binGroupingX=cms.untracked.int32(1),
    binGroupingY=cms.untracked.int32(1),
    fiducialXLow=cms.untracked.vdouble(fiducialXLow),
    fiducialYLow=cms.untracked.vdouble(fiducialYLow),
    fiducialYHigh=cms.untracked.vdouble(fiducialYHigh),
    # producerTag=cms.untracked.string("CTPPSTestProtonReconstruction"),
    producerTag=cms.untracked.string(""), #TODO: should deal with different producer tags  
    # ex.  "ctppsPixelLocalTracks"     ""           "RECO" (phisics dataset)
    # ex. "ctppsPixelLocalTracksAlCaRecoProducer"   ""                "ALCARECO"  (alcaPPS)

    detectorTiltAngle=cms.untracked.double(18.4),
    detectorRotationAngle=cms.untracked.double(-8),

    #FOR INTERPORT EFFICIENCY
    maxChi2Prob=cms.untracked.double(options.maxChi2Prob),
    maxTracksInProbePot=cms.untracked.int32(options.maxTracksInProbePot),    
    minTracksInProbePot=cms.untracked.int32(options.minTracksInProbePot),    
    maxTracksInTagPot=cms.untracked.int32(options.maxTracksInTagPot),    
    minTracksInTagPot=cms.untracked.int32(options.minTracksInTagPot),  
    recoInfo=cms.untracked.int32(options.recoInfo),
    debug=cms.untracked.bool(True),
)


#SETUP OUTPUT
process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string(options.outputFileName)
)


#SCHEDULE JOB
process.path = cms.Path(
    process.worker
)

process.end_path = cms.EndPath(
    process.dqmOutput
)

process.schedule = cms.Schedule(
    process.path,
    process.end_path
)