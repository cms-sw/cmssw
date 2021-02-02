import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000000                            # do not clog output with IO

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000000) )       # large number of events is needed since we probe 5000LS for run (see below)

####################################################################
# Empty source 
####################################################################

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(273291),
                            firstLuminosityBlock = cms.untracked.uint32(1),           # probe one LS after the other
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
from CondCore.CondDB.CondDB_cfi import *
CondDBBeamSpotObjects = CondDB.clone(connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'))
process.dbInput = cms.ESSource("PoolDBESSource",
                               CondDBBeamSpotObjects,
                               toGet = cms.VPSet(cms.PSet(record = cms.string('BeamSpotObjectsRcd'),
                                                          tag = cms.string('BeamSpotObjects_PCL_byLumi_v0_prompt') #choose your own favourite
                                                          )
                                                 )
                               )

####################################################################
# Load and configure analyzer
####################################################################
process.load("CondTools.BeamSpot.BeamSpotRcdRead_cfi")
process.BeamSpotRead.rawFileName = 'reference_prompt_BeamSpotObjects_PCL_byLumi_v0_prompt.txt'

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("reference_prompt_BeamSpotObjects_2016_LumiBased_v0_offline.root")
                                   ) 

# Put module in path:
process.p = cms.Path(process.BeamSpotRead)
