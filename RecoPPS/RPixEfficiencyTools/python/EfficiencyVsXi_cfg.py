import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("Demo")

import FWCore.ParameterSet.VarParsing as VarParsing
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
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
options.register('era',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "PPS era to be analyzed")
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
options.register('useMultiRPEfficiency',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.bool,
                "Use multiRP efficiency instead of refinedEfficiencyMap")
options.register('useInterPotEfficiency',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.bool,
                "Use interpot efficiency instead of refinedEfficiencyMap")
options.register('useMultiRPProtons',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.bool,
                "Use multiRP protons")
options.useInterPotEfficiency = False
options.useMultiRPEfficiency = False
options.useMultiRPProtons = False
options.parseArguments()

if (options.useInterPotEfficiency & options.useMultiRPEfficiency):
    print("ERROR: useMultiRPEfficiency and useInterPotEfficiency cannot be true at the same time!")
    sys.exit(1)

import FWCore.Utilities.FileUtils as FileUtils
fileList = FileUtils.loadListFromFile (options.sourceFileList) 
inputFiles = cms.untracked.vstring( *fileList)

process.load("FWCore.MessageService.MessageLogger_cfi")

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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(3000000) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = inputFiles,    
    skipBadFiles = cms.untracked.bool(True),
)


if options.useJsonFile == True:
    print("Using JSON file...")
    import FWCore.PythonUtilities.LumiList as LumiList
    if options.jsonFileName == '':
        jsonFileName = 'test/JSONFiles/Run'+str(options.runNumber)+'.json'
    else:
        jsonFileName = options.jsonFileName
    print(jsonFileName)
    process.source.lumisToProcess = LumiList.LumiList(filename = jsonFileName).getVLuminosityBlockRange()

# Fiducial region for tracks
# RP order 0_0, 0_2, 1_0, 1_2 at the top left angle of the RP track map (for tilted pots)
# cuts for dps note
# fiducialXLow = [2.85,2.28,3.28,2.42]
# fiducialXHigh = [99,99,99,99]
# fiducialYLow = [-11.5,-10.9,-11.6,-10.3]
# fiducialYHigh = [3.8,4.4,3.7,5.2]

# cuts for POG plots
# fiducialXLow = [0,0,0,0]
# fiducialXHigh = [99,99,99,99]

if options.era == 'A':
    fiducialXLow = [2.71,2.278,3.000,2.420]
    fiducialXHigh = [17.927,24.62,18.498,25.045]
    fiducialYLow = [-11.598,-10.898,-11.298,-10.398]
    fiducialYHigh = [3.698,4.398,4.098,5.098]
    print('Era '+options.era+' cuts set')
elif options.era == 'B1':
    fiducialXLow = [2.850,2.420,3.000,2.420]
    fiducialXHigh = [17.927,24.62,18.07,25.045]
    fiducialYLow = [-11.598,-10.798,-11.198,-10.398]
    fiducialYHigh = [3.698,4.298,4.098,5.098]
    print('Era '+options.era+' cuts set')
elif options.era == 'B2':
    fiducialXLow = [2.562,2.135,3.000,2.279]
    fiducialXHigh = [17.64,24.47,17.931,24.76]
    fiducialYLow = [-11.098,-11.398,-10.498,-10.598]
    fiducialYHigh = [4.198,3.798,4.698,4.498]
    print('Era '+options.era+' cuts set')
elif options.era == 'C':
    fiducialXLow = [2.564,2.278,3.00,2.279]
    fiducialXHigh = [17.93,24.62,17.931,24.76]
    fiducialYLow = [-11.098,-11.398,-10.498,-10.598]
    fiducialYHigh = [4.198,3.698,4.698,4.398]
    print('Era '+options.era+' cuts set')
elif options.era == 'D1':
    fiducialXLow = [2.847,2.278,3.000,2.279]
    fiducialXHigh = [17.93,24.62,17.931,24.76]
    fiducialYLow = [-11.098,-11.398,-10.498,-10.598]
    fiducialYHigh = [4.098,3.698,4.698,4.398]
    print('Era '+options.era+' cuts set')
elif options.era == 'D2':
    fiducialXLow = [2.847,2.278,3.000,2.279]
    fiducialXHigh = [17.931,24.62,17.931,24.76]
    fiducialYLow = [-10.598,-11.598,-9.998,-10.598]
    fiducialYHigh = [4.498,3.398,4.698,3.898]
elif options.era != '':
    print('Era not found! Exiting...')
    sys.exit(1)

# no cuts
# fiducialXLow = [0,0,0,0]
# fiducialXHigh = [99,99,99,99]
# fiducialYLow = [-99.,-99.,-99.,-99.]
# fiducialYHigh = [99.,99.,99.,99.]

firstRunOfTheYear = 314247
lastRunPreTs1     = 317696
lastRunPreTs2     = 322633
lastRunOfTheYear  = 324897

runNumber=options.runNumber
if runNumber < firstRunOfTheYear:
    print("This run belongs to before 2018 data taking")
elif runNumber <= lastRunPreTs1:
    print("Analyzing Pre-TS1 data")
elif runNumber <= lastRunPreTs2:
    print("Analyzing data taken between TS1 and TS2")
    # for i in range(4):
    #     if (i == 1 or i == 3):
    #         fiducialYLow[i] -= 0.5
    #         fiducialYHigh[i] -= 0.5
    #     else:
    #         fiducialYLow[i] += 0.5
    #         fiducialYHigh[i] += 0.5
elif runNumber <= lastRunOfTheYear:
    print("Analyzing Post-TS2 data")
    # for i in range(4):
        # if (i == 1 or i == 3):
        #     fiducialYLow[i] -= 1
        #     fiducialYHigh[i] -= 1
        # else:
        #     fiducialYLow[i] += 1
        #     fiducialYHigh[i] += 1
elif runNumber > lastRunOfTheYear:
    print("This run doesn't belong to 2018 data taking")
    
process.demo = cms.EDAnalyzer('EfficiencyVsXi_2018',
    # outputFileName=cms.untracked.string("RPixAnalysis_RecoLocalTrack_ReferenceRunAfterTS2.root"),
    efficiencyFileName=cms.untracked.string(options.efficiencyFileName),
    outputFileName=cms.untracked.string(options.outputFileName),
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
    useMultiRPEfficiency=cms.untracked.bool(options.useMultiRPEfficiency),
    useInterPotEfficiency=cms.untracked.bool(options.useInterPotEfficiency),
    useMultiRPProtons=cms.untracked.bool(options.useMultiRPProtons),
    producerTag=cms.untracked.string("ReMiniAOD")
)

process.p = cms.Path(process.demo)