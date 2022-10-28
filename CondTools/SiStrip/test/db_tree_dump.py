#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from fnmatch import fnmatch
from pdb import set_trace

#################################################
options = VarParsing.VarParsing("analysis")

options.register ('outputRootFile',
                  "",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,         # string, int, or float
                  "output root file")
options.register ('records',
                  [],
                  VarParsing.VarParsing.multiplicity.list, # singleton or list
                  VarParsing.VarParsing.varType.string,    # string, int, or float
                  "record:tag names to be used/changed from GT")
options.register ('external',
                  [],
                  VarParsing.VarParsing.multiplicity.list, # singleton or list
                  VarParsing.VarParsing.varType.string,    # string, int, or float
                  "record:fle.db picks the following record from this external file")
options.register ('runNumber',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,            # string, int, or float
                  "run number")
options.register ('runStartTime',
                  1,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,            # string, int, or float
                  "run start time")
options.register ('GlobalTag',
                  '',
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,         # string, int, or float
                  "Correct noise for APV gain?")

options.parseArguments()

process = cms.Process("Reader")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff') 

###################################################################
# Messages
###################################################################
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiStripDB2Tree=dict()
process.MessageLogger.RecordInfo=dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiStripDB2Tree = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    RecordInfo     = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    #input = cms.untracked.int32(-1) ### shall be -1 for the EmptyIOVSource
    )

# process.source = cms.Source("EmptyIOVSource",
#                             firstValue = cms.uint64(options.runNumber),
#                             lastValue = cms.uint64(options.runNumber),
#                             timetype = cms.string('runnumber'),
#                             interval = cms.uint64(1)
#                             )

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.runNumber),
                            numberEventsInRun = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),
                            firstTime = cms.untracked.uint64(options.runStartTime),
                            timeBetweenEvents = cms.untracked.uint64(1)
                            )

connection_map = [
    ('SiStrip*', 'frontier://PromptProd/CMS_CONDITIONS'),
    ]
if options.external:
    connection_map.extend(
        (i.split(':')[0], 'sqlite_file:%s' % i.split(':')[1]) for i in options.external
        )

connection_map.sort(key=lambda x: -1*len(x[0]))
def best_match(rcd):
    print(rcd)
    for pattern, string in connection_map:
        print(pattern, fnmatch(rcd, pattern))
        if fnmatch(rcd, pattern):
            return string
records = []
if options.records:
    for record in options.records:
        rcd, tag = tuple(record.split(':'))
        records.append(
            cms.PSet(
                record = cms.string(rcd),
                tag    = cms.string(tag),
                connect = cms.untracked.string(best_match(rcd))
                )
            )

if options.GlobalTag:
    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
    from Configuration.AlCa.GlobalTag import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag, options.GlobalTag, '')
    print("using global tag: %s" % process.GlobalTag.globaltag.value())
    #process.GlobalTag.DumpStat = cms.untracked.bool(True)  #to dump what records have been accessed
    process.GlobalTag.toGet = cms.VPSet(*records)
else:
    print("overriding using local conditions: %s" %records)
    process.poolDBESSource = cms.ESSource(
        "PoolDBESSource",
        BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
        DBParameters = cms.PSet(
            messageLevel = cms.untracked.int32(1),  # it used to be 2
            authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
            ),
        #DumpStat = cms.untracked.bool(True),
        timetype = cms.untracked.string('runnumber'),
        toGet = cms.VPSet(records),
        connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS')
        )


from CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi import*
siStripQualityESProducer.ListOfRecordToMerge = cms.VPSet(
    cms.PSet(record = cms.string("SiStripDetVOffRcd"), tag = cms.string('')),    # DCS information
    cms.PSet(record = cms.string('SiStripDetCablingRcd'), tag = cms.string('')), # Use Detector cabling information to exclude detectors not connected            
    cms.PSet(record = cms.string('SiStripBadChannelRcd'), tag = cms.string('')), # Online Bad components
    cms.PSet(record = cms.string('SiStripBadFiberRcd'), tag = cms.string('')),   # Bad Channel list from the selected IOV as done at PCL
    cms.PSet(record = cms.string('RunInfoRcd'), tag = cms.string(''))            # List of FEDs exluded during data taking          
    )

siStripQualityESProducer.ReduceGranularity = cms.bool(False)
siStripQualityESProducer.ThresholdForReducedGranularity = cms.double(0.3)
siStripQualityESProducer.appendToDataLabel = 'MergedBadComponent'
siStripQualityESProducer.PrintDebugOutput = cms.bool(True)

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(options.outputRootFile)
)
process.treeDump = cms.EDAnalyzer('SiStripDB2Tree',
                                  StripQualityLabel = cms.string('MergedBadComponent') 
                                  )
process.p = cms.Path(process.treeDump)
