import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import copy, sys, os

process = cms.Process("Misaligner")

###################################################################
# Setup 'standard' options
###################################################################
options = VarParsing.VarParsing()

options.register('myScenario',
                 "MisalignmentScenario_PhaseI_PseudoAsymptotic", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "scenario to apply")

options.register('mySigma',
                 -1, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.float, # string, int, or float
                 "sigma for random misalignment in um")

options.register('inputDB',
                 None, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "input database file to override GT (optional)")

options.parseArguments()

###################################################################
# Message logger service
###################################################################
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10000000)
    )
)
# replace MessageLogger.debugModules = { "*" }
# service = Tracer {}

###################################################################
# Ideal geometry producer and standard includes
###################################################################
process.load('Configuration.Geometry.GeometryRecoDB_cff')

###################################################################
# Just state the Global Tag (and pick some run)
###################################################################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag, "auto:run2_design", "")
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase1_2017_design", "")
print "Using global tag:", process.GlobalTag.globaltag.value()

###################################################################
# This uses the object from the tag and applies the misalignment scenario on top of that object
###################################################################
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")
process.AlignmentProducer.doMisalignmentScenario=True
process.AlignmentProducer.applyDbAlignment=True
process.AlignmentProducer.checkDbAlignmentValidity=False #otherwise error thrown for IOV dependent GTs
import Alignment.TrackerAlignment.Scenarios_cff as scenarios

if hasattr(scenarios, options.myScenario):
    print "Using scenario:", options.myScenario
    print "    with sigma:", options.mySigma
    print
    process.AlignmentProducer.MisalignmentScenario = getattr(scenarios, options.myScenario)
else:
    print "----- Begin Fatal Exception -----------------------------------------------"
    print "Unrecognized",options.myScenario,"misalignment scenario !!!!"
    print "Aborting cmsRun now, please check your input"
    print "----- End Fatal Exception -------------------------------------------------"
    sys.exit(1)

sigma = options.mySigma
if sigma > 0:
    process.AlignmentProducer.MisalignmentScenario.scale = cms.double(0.0001*sigma) # shifts are in cm

if options.inputDB is not None:
    process.GlobalTag.toGet.extend([
            cms.PSet(
                connect = cms.string("sqlite_file:"+options.inputDB),
                record = cms.string("TrackerAlignmentRcd"),
                tag = cms.string("Alignments")
                ),
            cms.PSet(
                connect = cms.string("sqlite_file:"+options.inputDB),
                record = cms.string("TrackerAlignmentErrorExtendedRcd"),
                tag = cms.string("AlignmentErrorsExtended")
                )
            ])

process.AlignmentProducer.saveToDB=True
process.AlignmentProducer.saveApeToDB=True

###################################################################
# Output name
###################################################################
outputfilename = None
scenariolabel = str(options.myScenario)
if sigma > 0:
    scenariolabel = scenariolabel+str(sigma)
outputfilename = "geometry_"+str(scenariolabel)+"__from_"
if options.inputDB is None:
    outputfilename += process.GlobalTag.globaltag.value()+".db"
else:
    outputfilename += options.inputDB

###################################################################
# Source
###################################################################
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

###################################################################
# Database output service
###################################################################
from CondCore.CondDB.CondDB_cfi import *
process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    CondDB,
    timetype = cms.untracked.string("runnumber"),
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string("TrackerAlignmentRcd"),
            tag = cms.string("Alignments")
            ),
        cms.PSet(
            record = cms.string("TrackerAlignmentErrorExtendedRcd"),
            tag = cms.string("AlignmentErrorsExtended")
            ),
        )
    )
process.PoolDBOutputService.connect = "sqlite_file:"+outputfilename
process.PoolDBOutputService.DBParameters.messageLevel = 2
