import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.cerr.FwkReport.reportEvery = 1000000                            # do not clog output with IO

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )       # large number of events is needed since we probe 5000LS for run (see below)

####################################################################
# Empty source 
####################################################################

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(1),
                            firstLuminosityBlock = cms.untracked.uint32(1),           # probe one LS after the other
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),  # probe one event per LS
                            numberEventsInRun = cms.untracked.uint32(1),              # a number of events > the number of LS possible in a real run (5000 s ~ 32 h)
                            )

####################################################################
# Connect to conditions DB
####################################################################

# either from Global Tag
# process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
# from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag,"auto:phase1_2023_realistic")

# ...or specify database connection and tag...
# from CondCore.CondDB.CondDB_cfi import *
# CondDBSimBeamSpot = CondDB.clone(connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'))
# process.dbInput = cms.ESSource("PoolDBESSource",
#                                CondDBSimBeamSpot,
#                                toGet = cms.VPSet(cms.PSet(record = cms.string('SimBeamSpotObjectsRcd'),
#                                                           tag = cms.string('your_tag_name')  # customize with input tag name
#                                                           )
#                                                  )
#                                )

# ...or specify local db file:
from CondCore.CondDB.CondDB_cfi import *
CondDBSimBeamSpot = CondDB.clone(connect = cms.string("sqlite_file:your_db_file.db")) # customize with input db file
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    CondDBSimBeamSpot,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SimBeamSpotObjectsRcd'),
        tag = cms.string('your_tag_name')  # customize with input tag name
    ))
)


####################################################################
# Load and configure analyzer
####################################################################
process.load("CondTools.BeamSpot.BeamProfile2DBRead_cfi")
process.BeamProfile2DBRead.rawFileName = 'reference_SimBeamSpotObjects.txt'

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("reference_SimBeamSpotObjects.root")
                                   ) 

# Put module in path:
process.p = cms.Path(process.BeamProfile2DBRead)
