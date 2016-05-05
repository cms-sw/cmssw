import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

###################################################################
# Setup 'standard' options
###################################################################
options = VarParsing.VarParsing()

options.register('conditionGT',
                 "auto:run2_data",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "condition global tag for the job (\"auto:run2_data\" is default)")

options.register('conditionOverwrite',
                 "",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "configuration to overwrite the condition into the GT (\"\" is default)")

options.register('inputCollection',
                 "ALCARECOSiStripCalMinBias",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "collections to be used for input (\"ALCARECOSiStripCalMinBias\" is default)")

options.register('outputFile',
                 "calibTreeTest.root",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "name for the output root file (\"calibTreeTest.root\" is default)")

options.register('inputFiles',
                 "/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalMinBias-16Dec2015-v1/60007/869EE593-1FAB-E511-AF99-0025905A60B4.root",
#                  '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalMinBias-16Dec2015-v1/60009/0C35C6BF-D3AA-E511-9BC9-0CC47A4C8E16.root',
#                  '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalMinBias-16Dec2015-v1/60009/38B847F9-05AA-E511-AB78-00259074AE82.root',
#                  '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalMinBias-16Dec2015-v1/60009/D0BAD20B-09AB-E511-B073-0026189438F6.root',
#                  '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalMinBias-16Dec2015-v1/60009/DEFA8704-CCAA-E511-8203-0CC47A4D7634.root',
#                  '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalMinBias-16Dec2015-v1/60009/FE24690A-2DAA-E511-A96A-00259074AE3E.root',
                 VarParsing.VarParsing.multiplicity.list,
                 VarParsing.VarParsing.varType.string,
                 "file to process")

options.register('maxEvents',
                 -1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of events to process (\"-1\" for all)")




# To use the prompt reco dataset please use 'generalTracks' as inputCollection
# To use the cosmic reco dataset please use 'ctfWithMaterialTracksP5' as inputCollection



options.parseArguments()

print "conditionGT       : ", options.conditionGT
print "conditionOverwrite: ", options.conditionOverwrite
print "inputCollection   : ", options.inputCollection
print "maxEvents         : ", options.maxEvents
print "outputFile        : ", options.outputFile
print "inputFiles        : ", options.inputFiles



process = cms.Process('CALIB')
process.load('CalibTracker.Configuration.setupCalibrationTree_cff')
process.load('Configuration/StandardSequences/MagneticField_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.conditionGT, options.conditionOverwrite)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.add_( cms.Service( "TFileService",
                           fileName = cms.string( options.outputFile ),
                           closeFileFast = cms.untracked.bool(True)  ) )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )
#process.source = cms.Source (
#    "PoolSource",
#    fileNames = cms.untracked.vstring(options.inputFiles)
#    )

#import runs
process.source = cms.Source (
  "PoolSource",
  fileNames = cms.untracked.vstring( options.inputFiles )
    )


process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

#definition of input collection
process.CalibrationTracks.src = 'ALCARECOSiStripCalMinBias' #cms.InputTag( options.inputCollection )
process.shallowTracks.Tracks  = 'ALCARECOSiStripCalMinBias' #cms.InputTag( options.inputCollection )
#process.shallowGainCalibrationAllBunch   = 'ALCARECOSiStripCalMinBias' #cms.InputTag( options.inputCollection )
#process.shallowGainCalibrationAllBunch0T = 'ALCARECOSiStripCalMinBias' #cms.InputTag( options.inputCollection )


# BSCNoBeamHalo selection (Not to use for Cosmic Runs) --- OUTDATED!!!
## process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
## process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

## process.L1T1=process.hltLevel1GTSeed.clone()
## process.L1T1.L1TechTriggerSeeding = cms.bool(True)
## process.L1T1.L1SeedsLogicalExpression = cms.string('(40 OR 41) AND NOT (36 OR 37 OR 38 OR 39)')

#process.TkCalPath = cms.Path(process.L1T1*process.TkCalFullSequence)
process.TkCalPath_StdBunch   = cms.Path(process.TkCalSeq_StdBunch)
process.TkCalPath_StdBunch0T = cms.Path(process.TkCalSeq_StdBunch0T)
process.TkCalPath_IsoMuon    = cms.Path(process.TkCalSeq_IsoMuon)
process.TkCalPath_IsoMuon0T  = cms.Path(process.TkCalSeq_IsoMuon0T)
process.TkCalPath_AagBunch   = cms.Path(process.TkCalSeq_AagBunch)
process.TkCalPath_AagBunch0T = cms.Path(process.TkCalSeq_AagBunch0T)


process.schedule = cms.Schedule( process.TkCalPath_StdBunch, 
                                 process.TkCalPath_StdBunch0T,
                                 process.TkCalPath_IsoMuon,
                                 process.TkCalPath_IsoMuon0T,
                                 process.TkCalPath_AagBunch,
                                 process.TkCalPath_AagBunch0T,
                               )


process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('OtherCMS', 
        'StdException', 
        'Unknown', 
        'BadAlloc', 
        'BadExceptionType', 
        'ProductNotFound', 
        'DictionaryNotFound', 
        'InsertFailure', 
        'Configuration', 
        'LogicError', 
        'UnimplementedFeature', 
        'InvalidReference', 
        'NullPointerError', 
        'NoProductSpecified', 
        'EventTimeout', 
        'EventCorruption', 
        'ScheduleExecutionFailure', 
        'EventProcessorFailure', 
        'FileInPathError', 
        'FileOpenError', 
        'FileReadError', 
        'FatalRootError', 
        'MismatchedInputFiles', 
        'ProductDoesNotSupportViews', 
        'ProductDoesNotSupportPtr', 
        'NotFound')
)
