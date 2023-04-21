import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from Configuration.AlCa.GlobalTag import GlobalTag
import FWCore.ParameterSet.VarParsing as VarParsing

#GLOBAL CONSTANT VARIABLES
# fiducial variables restrict the area to analyze 
# - the current parameters cover the whole possible area 
fiducialXLow = [0,0,0,0]
fiducialXHigh = [99,99,99,99]
fiducialYLow = [-99.,-99.,-99.,-99.]
fiducialYHigh = [99.,99.,99.,99.]

#SETUP PROCESS
process = cms.Process("Demo")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    FailPath = cms.untracked.vstring('Type Mismatch')
    )
options = VarParsing.VarParsing ()
options.register('outputFileName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "output ROOT file name")
options.register('efficiencyFileName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "efficiency ROOT file name - output of the EfficiencyTool_2018")
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
options.parseArguments()

import FWCore.Utilities.FileUtils as FileUtils
fileList = FileUtils.loadListFromFile (options.sourceFileList) 
inputFiles = cms.untracked.vstring( *fileList)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQM.Integration.config.environment_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2018_cfi")

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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) ) 

if options.useJsonFile == True:
    print("Using JSON file...")
    import FWCore.PythonUtilities.LumiList as LumiList
    if options.jsonFileName == '':
        jsonFileName = 'test/JSONFiles/Run'+str(options.runNumber)+'.json'
    else:
        jsonFileName = options.jsonFileName
    print(jsonFileName)
    process.source.lumisToProcess = LumiList.LumiList(filename = jsonFileName).getVLuminosityBlockRange()


runNumber=options.runNumber

#SETUP GLOBAL TAG
process.GlobalTag = GlobalTag(process.GlobalTag, '123X_dataRun2_v4')


#SETUP INPUT
process.source = cms.Source("PoolSource",
    fileNames = inputFiles
)

#SETUP OUTPUT
process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string(options.outputFileName)
)

process.worker = DQMEDAnalyzer('ReferenceAnalysisDQMWorker',
    outputFileName=cms.untracked.string(options.outputFileName),
    efficiencyFileName=cms.untracked.string(options.efficiencyFileName),
    minNumberOfPlanesForEfficiency=cms.int32(3),
    minNumberOfPlanesForTrack=cms.int32(3),
    minTracksPerEvent=cms.int32(0),
    maxTracksPerEvent=cms.int32(99),
    binGroupingX=cms.untracked.int32(1),
    binGroupingY=cms.untracked.int32(1),
    fiducialXLow=cms.untracked.vdouble(fiducialXLow),
    fiducialXHigh=cms.untracked.vdouble(fiducialXHigh),
    fiducialYLow=cms.untracked.vdouble(fiducialYLow),
    fiducialYHigh=cms.untracked.vdouble(fiducialYHigh),
    producerTag=cms.untracked.string("CTPPSTestProtonReconstruction"),
    detectorTiltAngle=cms.untracked.double(18.4),
    detectorRotationAngle=cms.untracked.double(-8),
    useMultiRPEfficiency=cms.untracked.bool(False),
    useMultiRPProtons=cms.untracked.bool(False),
    useInterPotEfficiency=cms.untracked.bool(False)
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

