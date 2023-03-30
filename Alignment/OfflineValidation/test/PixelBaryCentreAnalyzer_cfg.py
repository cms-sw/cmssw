import FWCore.ParameterSet.Config as cms

process = cms.Process("READ")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register('lumisPerRun',
                1,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "the number of lumis to be processed per-run.")
options.register('firstRun',
                290550,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "the first run number be processed")
options.register('lastRun',
                325175,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "the run number to stop")
options.register('unitTest',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "is it a unit test?")

options.parseArguments()

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = options.lumisPerRun*1000   # do not clog output with I/O

if options.unitTest:
    numberOfRuns = 10
else:
    numberOfRuns = options.lastRun - options.firstRun + 1
print("number of Runs "+str(numberOfRuns))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.lumisPerRun*numberOfRuns) ) 

####################################################################
# Empty source 
####################################################################
#import FWCore.PythonUtilities.LumiList as LumiList
#DCSJson='/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/DCSOnly/json_DCSONLY.txt'

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.firstRun),
                            firstLuminosityBlock = cms.untracked.uint32(1),           # probe one LS after the other
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),  # probe one event per LS
                            numberEventsInRun = cms.untracked.uint32(options.lumisPerRun),           # a number of events > the number of LS possible in a real run (5000 s ~ 32 h)
                            )

####################################################################
# Connect to conditions DB
####################################################################

# either from Global Tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,"auto:run2_data")

tkAligns = {"prompt":"TrackerAlignment_PCL_byRun_v2_express","EOY":"TrackerAlignment_v24_offline","rereco":"TrackerAlignment_v29_offline"}

for label in tkAligns.keys() :

   process.GlobalTag.toGet.append(
     cms.PSet(
       record = cms.string("TrackerAlignmentRcd"),
       label = cms.untracked.string(label),
       tag = cms.string(tkAligns[label]),
       connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
     )
   )


beamSpots = {"prompt":"BeamSpotObjects_PCL_byLumi_v0_prompt","rereco":"BeamSpotObjects_2016_2017_2018UL_SpecialRuns_LumiBased_v1"}

for label in beamSpots.keys() :

    process.GlobalTag.toGet.append(
      cms.PSet(
        record = cms.string("BeamSpotObjectsRcd"),
        label = cms.untracked.string(label),
        tag = cms.string(beamSpots[label]),
        connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
      )
    )

# ...or specify database connection and tag:  
#from CondCore.CondDB.CondDB_cfi import *
#CondDBBeamSpotObjects = CondDB.clone(connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'))
#process.dbInput = cms.ESSource("PoolDBESSource",
#                               CondDBBeamSpotObjects,
#                               toGet = cms.VPSet(cms.PSet(record = cms.string('BeamSpotObjectsRcd'),
#                                                          tag = cms.string('BeamSpotObjects_PCL_byLumi_v0_prompt') #choose your own favourite
#                                                          )
#                                                 )
#                               )

####################################################################
# Load and configure analyzer
####################################################################
bcLabels_ = [] # cms.untracked.vstring("")
bsLabels_ = [] # cms.untracked.vstring("")

for label in tkAligns.keys() :
    bcLabels_.append(label)

for label in beamSpots.keys() :
    bsLabels_.append(label)

from Alignment.OfflineValidation.pixelBaryCentreAnalyzer_cfi import pixelBaryCentreAnalyzer as _pixelBaryCentreAnalyzer

process.PixelBaryCentreAnalyzer = _pixelBaryCentreAnalyzer.clone(
    usePixelQuality = False,
    tkAlignLabels = bcLabels_,
    beamSpotLabels = bsLabels_
)

process.PixelBaryCentreAnalyzerWithPixelQuality = _pixelBaryCentreAnalyzer.clone(
    usePixelQuality = True,
    tkAlignLabels = bcLabels_,
    beamSpotLabels = bsLabels_
)

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("PixelBaryCentre_2017.root")
                                   ) 

# Put module in path:
process.p = cms.Path(process.PixelBaryCentreAnalyzer*process.PixelBaryCentreAnalyzerWithPixelQuality)
