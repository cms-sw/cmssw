import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.AlignPCLThresholdsReader=dict()  
process.MessageLogger.AlignPCLThresholds=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    AlignPCLThresholdsReader = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    AlignPCLThresholds       = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    AlignPCLThresholdsHG     = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )

##
## Var Parsing
##
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register('readLGpayload',
                False,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.bool,
                "Read old payload type used for LG thresholds")
options.parseArguments()

##
## Define record, class and module based on option
##
[rcdName, className, moduleName] = ["AlignPCLThresholdsRcd","AlignPCLThresholds","AlignPCLThresholdsLGReader"] \
                                   if (options.readLGpayload) else ["AlignPCLThresholdsHGRcd","AlignPCLThresholdsHG","AlignPCLThresholdsHGReader"]

print("using %s %s %s" % (rcdName, className, moduleName))
##
## Empty Source
##
process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )
##
## Get the payload
##
from CondCore.CondDB.CondDB_cfi import *
inputDBfile = 'sqlite_file:mythresholds_%s.db' % ("LG" if(options.readLGpayload) else "HG") 

CondDBThresholds = CondDB.clone(connect = cms.string(inputDBfile))

process.dbInput = cms.ESSource("PoolDBESSource",
                               CondDBThresholds,
                               toGet = cms.VPSet(cms.PSet(record = cms.string(rcdName),
                                                          tag = cms.string('PCLThresholds_express_v0') # choose tag you want
                                                          )
                                                 )
                               )
##
## Retrieve it and check it's available in the ES
##
process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
                             toGet = cms.VPSet(cms.PSet(record = cms.string(rcdName),
                                                        data = cms.vstring(className)
                                                        )
                                               ),
                             verbose = cms.untracked.bool(True)
                             )

##
## Read it back
##
from CondFormats.PCLConfig.edmtestAlignPCLThresholdsReaderAlignPCLThresholdsAlignPCLThresholdsRcd_cfi import *
from CondFormats.PCLConfig.edmtestAlignPCLThresholdsReaderAlignPCLThresholdsHGAlignPCLThresholdsHGRcd_cfi import *

if(options.readLGpayload):
    process.ReadDB = edmtestAlignPCLThresholdsReaderAlignPCLThresholdsAlignPCLThresholdsRcd.clone(
        printDebug = True,
        outputFile = 'AlignPCLThresholds_%s.log' % ("LG" if(options.readLGpayload) else "HG")
    )
else:
    process.ReadDB = edmtestAlignPCLThresholdsReaderAlignPCLThresholdsHGAlignPCLThresholdsHGRcd.clone(
        printDebug = True,
        outputFile = 'AlignPCLThresholds_%s.log' % ("LG" if(options.readLGpayload) else "HG")
    )

process.p = cms.Path(process.get+process.ReadDB)
