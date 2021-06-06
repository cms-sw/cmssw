from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("Demo")

##
## prepare options
##
options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "103X_dataRun2_HLT_relval_v8",VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.register ('forHLT',
                  True,VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.bool,          # string, int, or float
                  "is for SiPixelGainCalibrationForHLT")

options.register ('firstRun',
                  1,VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "first run to be processed")

options.parseArguments()

##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPixelGainCalibScaler=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(100000)
                                   ),                                                      
    SiPixelGainCalibScaler = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )

process.load("Configuration.Geometry.GeometryRecoDB_cff") # Ideal geometry and interface 
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')

### Dirty trick to avoid geometry mismatches
process.trackerGeometryDB.applyAlignment = False

##
## Selects the output record
##
MyRecord=""
if options.forHLT:
    MyRecord="SiPixelGainCalibrationForHLTRcd"
else:
    MyRecord="SiPixelGainCalibrationOfflineRcd"

##
## Printing options
##
print("Using Global Tag:", process.GlobalTag.globaltag._value)
print("first run to be processed:",options.firstRun)
print("is for HLT? ","yes" if options.forHLT else "no!")
print("outputing on record: ",MyRecord)

##
## Empty Source
##
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(400000))      

####################################################################
# Empty source 
####################################################################
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.firstRun),
                            numberEventsInRun = cms.untracked.uint32(1),           # a number of events in single run 
                            )

####################################################################
# Analysis Module
####################################################################
process.demo = cms.EDAnalyzer('SiPixelGainCalibScaler',
                              isForHLT = cms.bool(options.forHLT),
                              record = cms.string(MyRecord),
                              parameters = cms.VPSet(
                                  cms.PSet(
                                      conversionFactor = cms.double(65.),
                                      conversionFactorL1 = cms.double(65.),
                                      offset = cms.double(-414.),
                                      offsetL1 = cms.double(-414.),
                                      phase = cms.uint32(0)
                                  ),
                                  cms.PSet(
                                      conversionFactor = cms.double(47.),
                                      conversionFactorL1 = cms.double(50.),
                                      offset = cms.double(-60.),
                                      offsetL1 = cms.double(-670.),
                                      phase = cms.uint32(1)
                                  )
                              )
                          )
##
## Database output service
##
process.load("CondCore.CondDB.CondDB_cfi")

##
## Output database (in this case local sqlite file)
##
process.CondDB.connect = 'sqlite_file:TEST_modifiedGains_'+process.GlobalTag.globaltag._value+("_HLTGain" if options.forHLT else "_offlineGain")+".db"
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string(MyRecord),
                                                                     tag = cms.string('scaledGains')
                                                                     )
                                                            )
                                          )

process.p = cms.Path(process.demo)
