import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import copy, sys, os

process = cms.Process("Misaligner")

###################################################################
# Setup 'standard' options
###################################################################
options = VarParsing.VarParsing()

options.register('myScenario',
                 "MisalignmentScenario_NonMisalignedBPIX", # default value
                 #~ "MisalignmentScenarioNoMisalignment", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "scenario to apply")

options.register('mySigma',
                 -1, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.float, # string, int, or float
                 "sigma for random misalignment in um")

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
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff') 
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_design', '')
print "Using global tag: %s" % process.GlobalTag.globaltag._value

###################################################################
# This uses the object from the tag and applies the misalignment scenario on top of that object
###################################################################
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")
process.AlignmentProducer.doMisalignmentScenario=True
process.AlignmentProducer.applyDbAlignment=True
process.AlignmentProducer.checkDbAlignmentValidity=False #otherwise error thrown for IOV dependent GTs
from Alignment.APEEstimation.MisalignmentScenarios_cff import *

isMatched = False
print "Using scenario :",options.myScenario
print "    with sigma :",options.mySigma

for objname,oid in globals().items():
    #print objname
    if (str(objname) == str(options.myScenario)):
        isMatched = True
        print "Using scenario:",objname
        process.AlignmentProducer.MisalignmentScenario = oid
   
if isMatched is not True:
    print "----- Begin Fatal Exception -----------------------------------------------"
    print "Unrecognized",options.myScenario,"misalignment scenario !!!!"
    print "Aborting cmsRun now, please check your input"
    print "----- End Fatal Exception -------------------------------------------------"
    os._exit(1)

sigma = options.mySigma
if sigma > 0:
    process.AlignmentProducer.MisalignmentScenario.scale = cms.double(0.0001*sigma) # shifts are in cm

process.AlignmentProducer.saveToDB=True
process.AlignmentProducer.saveApeToDB=False

###################################################################
# Output name
###################################################################
outputfilename = None
scenariolabel = str(options.myScenario)
if sigma > 0:
    scenariolabel = scenariolabel+str(sigma)
outputfilename = "geometry_"+str(scenariolabel)+"_from"+process.GlobalTag.globaltag._value.replace('::All','')+".db"

###################################################################
# Source
###################################################################
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

###################################################################
# Database output service
###################################################################

import CondCore.DBCommon.CondDBSetup_cfi
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
    # Writing to oracle needs the following shell variable setting (in zsh):
    # export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb
    # connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_ALIGNMENT'),  # preparation/develop. DB
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:'+outputfilename),                                      
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('TrackerAlignmentRcd'),
            tag = cms.string('Alignments')
        ), 
    )
)
process.PoolDBOutputService.DBParameters.messageLevel = 2
