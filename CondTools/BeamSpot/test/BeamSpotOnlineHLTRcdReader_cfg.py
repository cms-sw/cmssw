import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100000                         # do not clog output with IO

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000000) )   # large number of events is needed since we probe 5000LS for run (see below)

####################################################################
# Empty source 
####################################################################

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(306171),                  # Run in ../data/BeamFitResults_Run306171.txt
                            firstLuminosityBlock = cms.untracked.uint32(497),         # Lumi in ../data/BeamFitResults_Run306171.txt
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),  # probe one event per LS
                            numberEventsInRun = cms.untracked.uint32(5000),           # a number of events > the number of LS possible in a real run (5000 s ~ 32 h)
                            )

####################################################################
# Connect to conditions DB
####################################################################

# either from Global Tag
# process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
# from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag,"auto:run2_data")

# ...or specify database connection and tag:  
#from CondCore.CondDB.CondDB_cfi import *
#CondDBBeamSpotObjects = CondDB.clone(connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'))
#process.dbInput = cms.ESSource("PoolDBESSource",
#                               CondDBBeamSpotObjects,
#                               toGet = cms.VPSet(cms.PSet(record = cms.string('BeamSpotOnlineHLTObjectsRcd'), # BeamSpotOnlineHLT record
#                                                          tag = cms.string('BSHLT_tag')                       # choose your favourite tag
#                                                          )
#                                                 )
#                               )
# ...or from a local db file
# input database (in this case the local sqlite file)
from CondCore.CondDB.CondDB_cfi import *
CondDBBeamSpotOnlineHLT = CondDB.clone(connect = cms.string("sqlite_file:test.db")) # customize with input db file
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    CondDBBeamSpotOnlineHLT,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('BeamSpotOnlineHLTObjectsRcd'),  # BeamSpotOnlineHLT record
        tag = cms.string('BSHLT_tag')                        # choose your favourite tag
    ))
)

####################################################################
# Load and configure analyzer
####################################################################
process.beamspotonlinereader = cms.EDAnalyzer("BeamSpotOnlineHLTRcdReader",
                              rawFileName = cms.untracked.string("test.txt") # choose an output name
                              )

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("test.root") # choose an output name
                                   )

# Put module in path:
process.p = cms.Path(process.beamspotonlinereader)
