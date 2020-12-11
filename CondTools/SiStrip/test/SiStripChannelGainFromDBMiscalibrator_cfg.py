from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("Demo")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "auto:run2_data",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.register ('runNumber',
                  303014,
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,           # string, int, or float
                  "run number")

options.parseArguments()


##
## MessageLogger
##
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiStripChannelGainFromDBMiscalibrator=dict()  
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("WARNING"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiStripChannelGainFromDBMiscalibrator  = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )

process.load("Configuration.Geometry.GeometryRecoDB_cff") # Ideal geometry and interface 
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.globalTag, '')

print("Using Global Tag:", process.GlobalTag.globaltag._value)

##
## Empty Source
##
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.runNumber),
                            numberEventsInRun = cms.untracked.uint32(1),
                            )


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

##
## Example smearing configurations
##

##
## separately partition by partition
##
byParition = cms.VPSet(
    cms.PSet(partition = cms.string("TIB"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.1),
             smearFactor = cms.double(0.2)
             ),
    cms.PSet(partition = cms.string("TOB"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.2),
             smearFactor = cms.double(0.15)
             ),
    cms.PSet(partition = cms.string("TID"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.3),
             smearFactor = cms.double(0.10)
             ),
    cms.PSet(partition = cms.string("TEC"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.4),
             smearFactor = cms.double(0.05)
             )
    )

##
## whole Strip tracker
##

wholeTracker = cms.VPSet(
    cms.PSet(partition = cms.string("Tracker"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.15),
             smearFactor = cms.double(0.05)
             )
    )


##
## down the hierarchy (Tracker,Subdetector,Side,Layer(Wheel)
##

subsets =  cms.VPSet(
    cms.PSet(partition = cms.string("Tracker"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(0.65),
             smearFactor = cms.double(0.05)
             ),
    cms.PSet(partition = cms.string("TEC"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.15),
             smearFactor = cms.double(0.02)
             ),
    cms.PSet(partition = cms.string("TECP"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.35),
             smearFactor = cms.double(0.02)
             ),
    cms.PSet(partition = cms.string("TECP_9"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(True),
             scaleFactor = cms.double(1.55),
             smearFactor = cms.double(0.02)
             )
    )


# process.demo = cms.EDAnalyzer('SiStripChannelGainFromDBMiscalibrator',
#                               record = cms.untracked.string("SiStripApvGainRcd"),
#                               gainType = cms.untracked.uint32(1), #0 for G1, 1 for G2
#                               params = subsets, # as a cms.VPset
#                               saveMaps = cms.bool(True)      
#                               )

process.load("CondTools.SiStrip.scaleAndSmearSiStripGains_cfi")
process.scaleAndSmearSiStripGains.gainType = 1        # 0 for G1, 1 for G2
process.scaleAndSmearSiStripGains.params   = subsets  # as a cms.VPset

##
## Database output service
##
process.load("CondCore.CondDB.CondDB_cfi")

##
## Output database (in this case local sqlite file)
##
process.CondDB.connect = 'sqlite_file:modifiedGains_'+ process.GlobalTag.globaltag._value+'_IOV_'+str(options.runNumber)+".db"
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('SiStripApvGainRcd'),
                                                                     tag = cms.string('modifiedGains')
                                                                     )
                                                            )
                                          )

process.p = cms.Path(process.scaleAndSmearSiStripGains)
