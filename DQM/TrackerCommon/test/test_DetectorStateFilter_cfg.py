import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process('reRECO',Run3)

options = VarParsing.VarParsing('analysis')
options.register('globalTag',
                 "auto:run3_data_prompt", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "input file name")                
options.register('isStrip',
                 True, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "true filters on Strips, false filters on Pixels")
options.parseArguments()

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')

process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.DetectorStatusFilter=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1)
                                   ),                                                      
    DetectorStatusFilter = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
    )

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
                            #fileNames = cms.untracked.vstring("/store/express/Commissioning2021/ExpressCosmics/FEVT/Express-v1/000/343/498/00000/004179ae-ac29-438a-bd2d-ea98139c21a7.root") # default value
                            fileNames = cms.untracked.vstring(options.inputFiles)
                            )
# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)

process.OUT = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_onlineMetaDataDigis_*_*',
                                                                      'keep *_scalersRawToDigi_*_*'),
                               SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('analysis_step')),
                               fileName = cms.untracked.string(options.outputFile)
                               )

# the module to be tested
from DQM.TrackerCommon.TrackerFilterConfiguration_cfi import detectorStateFilter
process.SiPixelFilter = detectorStateFilter.clone(DetectorType = 'pixel',
                                                  DebugOn = True)

process.SiStripFilter = detectorStateFilter.clone(DetectorType = 'sistrip',
                                                  DebugOn = True)

#process.analysis_step = cms.Path(process.detectorStateFilter)
if(options.isStrip) :
    process.analysis_step = cms.Path(process.SiStripFilter)
else:
    process.analysis_step = cms.Path(process.SiPixelFilter)

# end path
process.printEventNumber = cms.OutputModule("AsciiOutputModule")
process.this_is_the_end = cms.EndPath(process.OUT*process.printEventNumber)

process.schedule = cms.Schedule(process.raw2digi_step,process.analysis_step,process.this_is_the_end)
