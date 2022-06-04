import FWCore.ParameterSet.Config as cms
import copy 

process = cms.Process("ProcessOne")

##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.AlignPCLThresholdsWriter=dict()  
process.MessageLogger.AlignPCLThresholds=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    AlignPCLThresholdsWriter = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    AlignPCLThresholds       = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    AlignPCLThresholdsHG     = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )

##
## Empty source
##
process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )
##
## Database output service
##
process.load("CondCore.CondDB.CondDB_cfi")

##
## Output database (in this case local sqlite file)
##
process.CondDB.connect = 'sqlite_file:mythresholds.db'
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('FooRcd'),
                                                                     tag = cms.string('PCLThresholds_express_v0')
                                                                     )
                                                            )
                                          )

##
## Impot the thresholds configuration
##
import CondFormats.PCLConfig.ThresholdsHG_cff as Thresholds

##
## Example on how to add to the default extra degrees of freedom
##
AddSurfaceThresholds = copy.deepcopy(Thresholds.default)
BPixSurface= cms.VPSet(
    cms.PSet(alignableId       = cms.string("TPBModule"),
             DOF               = cms.string("Surface1"),
             cut               = cms.double(0.1),
             sigCut            = cms.double(0.1),
             maxMoveCut        = cms.double(0.1),
             maxErrorCut       = cms.double(10.0)
             )
    )

DefaultPlusSurface = AddSurfaceThresholds+BPixSurface
#print DefaultPlusSurface.dumpPython()

process.WriteInDB = cms.EDAnalyzer("AlignPCLThresholdsWriter",
                                   record= cms.string('FooRcd'),
                                   ### minimum number of records found in pede output 
                                   minNRecords = cms.uint32(25000), 
                                   #thresholds  = cms.VPSet()         # empty object
                                   #thresholds = DefaultPlusSurface   # add extra deegree of freedom
                                   thresholds = Thresholds.default   # as a cms.VPset
                                   )

process.p = cms.Path(process.WriteInDB)
