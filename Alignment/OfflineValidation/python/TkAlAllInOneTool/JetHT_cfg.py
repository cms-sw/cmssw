'''
cfg to produce alignment validation plots from JetHT sample 
here doing refit of tracks and vertices using latest alignment 
'''

# Define the process
import FWCore.ParameterSet.Config as cms
process = cms.Process("JetHTAnalyzer")

# Choose whether to run with Monte Carlo or data settings based on command line argument
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register ('config',
                  "", # Default file
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,         # string, int, or float
                  "AllInOne config.")
options.parseArguments()

#Read in AllInOne config in JSON format
from Alignment.OfflineValidation.TkAlAllInOneTool.utils import _byteify
import json
import os

if options.config == "":
    configuration = {"validation": {},
                     "alignment": {}}
else:
    with open(options.config, "r") as configFile:
        configuration = _byteify(json.load(configFile, object_hook=_byteify), ignore_dicts=True)

# Read parameters from the configuration file
useMC = configuration["validation"].get("mc", False)
printTriggers = configuration["validation"].get("printTriggers", False)
triggerFilter = str(configuration["validation"].get("triggerFilter", "nothing"))
iovListFile = str(configuration["validation"].get("iovListFile", "nothing"))
iovListList = configuration["validation"].get("iovList", [0,500000])
ptBorders = configuration["validation"].get("profilePtBorders", [3,5,10,20,50,100])
trackCollection = str(configuration["validation"].get("trackCollection", "ALCARECOTkAlMinBias"))
maxEventsToRun = configuration["validation"].get("maxevents", 1)

# The default global tag is suiteble for the unit test file for data
globalTag = str(configuration["alignment"].get("globaltag", "auto:run2_data"))

# Alignment conditions can be also loaded from a configuration file instead of database
alignmentFile = str(configuration["validation"].get("TrackerAlignmentRcdFile", "nothing"))
alignmentErrorFile = str(configuration["validation"].get("TrackerAlignmentErrorFile", "nothing"))


# If IOV list file is given, read the IOV:s from the file
if not (iovListFile == "nothing" or iovListFile == ""):

    # Expand CMSSW_BASE
    iovListFile = iovListFile.replace("CMSSW_BASE", os.environ["CMSSW_BASE"])

    # Read the file
    iovListList = []
    iovReader = open(iovListFile,"r")
    iovContents = iovReader.readlines()

    for line in iovContents:
        lineContents = line.split()
        iovListList.append(int(lineContents[0]))

    # Add a dummy run to the end of the list to make the last number its own IOV
    iovListList.append(iovListList[-1]+100)


###################################################################
# Messages
###################################################################
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)

# If maxEvents not specified, only do one event
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(maxEventsToRun) )

###################################################################
# Basic modules
###################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.Geometry.GeometryDB_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")

##Read filenames from given .txt file and define input source
readFiles = []

if "dataset" in configuration["validation"]:
    with open(configuration["validation"]["dataset"], "r") as datafiles:
        for fileName in datafiles.readlines():
            readFiles.append(fileName.replace("\n", ""))

    ##Define input source
    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(readFiles),
                                skipEvents = cms.untracked.uint32(0)
                               )

# If no text file with dataset given, read default file
else:
    print(">>>>>>>>>> JetHT_cfg.py: msg%-i: dataset not specified in configuration! Loading default file!")

    if useMC:
        print(">>>>>>>>>> JetHT_cfg.py: msg%-i: Default file for 2017 MC from 170-300 pT hat bin.")
        process.source = cms.Source("PoolSource",
                              fileNames = cms.untracked.vstring('root://xrootd-cms.infn.it//store/mc/RunIIWinter19PFCalibDRPremix/QCD_Pt_170to300_TuneCP5_13TeV_pythia8/ALCARECO/TkAlMinBias-2017Conditions_105X_mc2017_realistic_v5-v1/10000/A4C71578-282D-384C-AF19-9A685808EEB8.root')
                              )
    else:
        print(">>>>>>>>>> JetHT_cfg.py: msg%-i: Default file read from 2018D JetHT dataset.")
        process.source = cms.Source("PoolSource",
                              fileNames = cms.untracked.vstring('root://xrootd-cms.infn.it//store/data/Run2018D/JetHT/ALCARECO/TkAlMinBias-12Nov2019_UL2018-v4/00000/009085FF-603B-044D-8879-0599B156831D.root') # A file from 2018 JetHT file in DESY

                              )

####################################################################
# Global tag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globalTag)

####################################################################
# Alignment conditions
####################################################################

import CalibTracker.Configuration.Common.PoolDBESSource_cfi

# Determine tracker conditions if defined in the configuration
if "conditions" in configuration["alignment"]:
    from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource

    for condition in configuration["alignment"]["conditions"]:
        setattr(process, "conditionsIn{}".format(condition), poolDBESSource.clone(
             connect = cms.string(str(configuration["alignment"]["conditions"][condition]["connect"])),
             toGet = cms.VPSet(
                        cms.PSet(
                                 record = cms.string(str(condition)),
                                 tag = cms.string(str(configuration["alignment"]["conditions"][condition]["tag"]))
                        )
                     )
            )
        )

        setattr(process, "prefer_conditionsIn{}".format(condition), cms.ESPrefer("PoolDBESSource", "conditionsIn{}".format(condition)))

# If the reference to the database is not provided, check if a database file is provided
if not (alignmentFile == "nothing" or alignmentFile == ""):
  process.conditionsInTrackerAlignmentRcd = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
       connect = cms.string("sqlite_file:" + alignmentFile),
       toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentRcd'),
                                 tag = cms.string('Alignments')
                                 )
                        )
      )
  process.prefer_conditionsInTrackerAlignmentRcd = cms.ESPrefer("PoolDBESSource", "conditionsInTrackerAlignmentRcd")

# If the reference to the database is not provided, check if a database file is provided
if not (alignmentErrorFile == "nothing" or alignmentErrorFile == ""):
  process.conditionsInTrackerAlignmentErrorExtendedRcd = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone(
       connect = cms.string("sqlite_file:" + alignmentErrorFile),
       toGet = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentErrorExtendedRcd'),
                                 tag = cms.string('APEs')
                                 )
                        )
      )
  process.prefer_conditionsInTrackerAlignmentErrorExtendedRcd = cms.ESPrefer("PoolDBESSource", "conditionsInTrackerAlignmentErrorExtendedRcd")

####################################################################
# Track and primary vertex refit
####################################################################


# Setup track refitter
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
# remove the following lines if you run on RECO files
process.TrackRefitter.src = trackCollection
process.TrackRefitter.NavigationSchool = ""

## PV refit
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices 
process.offlinePrimaryVerticesFromRefittedTrks  = offlinePrimaryVertices.clone()
process.offlinePrimaryVerticesFromRefittedTrks.TrackLabel                                       = cms.InputTag("TrackRefitter") 
process.offlinePrimaryVerticesFromRefittedTrks.vertexCollections.maxDistanceToBeam              = 1
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxNormalizedChi2             = 20
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.minSiliconLayersWithHits      = 5
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.maxD0Significance             = 5.0 
process.offlinePrimaryVerticesFromRefittedTrks.TkFilterParameters.minPixelLayersWithHits        = 2   

###################################################################
# The trigger filter module
###################################################################
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
if not (triggerFilter == "nothing" or triggerFilter == ""):
    triggerPath = triggerFilter
    process.HLTFilter = triggerResultsFilter.clone(
        triggerConditions = cms.vstring(triggerPath),
        hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
        l1tResults = cms.InputTag( "" ),
        throw = cms.bool(False)
    )

###################################################################
# The analysis module
###################################################################
process.jetHTAnalyzer = cms.EDAnalyzer('JetHTAnalyzer',
                                       vtxCollection       = cms.InputTag("offlinePrimaryVerticesFromRefittedTrks"),
                                       trackCollection	   = cms.InputTag("TrackRefitter"),		
                                       triggerResults      = cms.InputTag("TriggerResults","","HLT"),
                                       printTriggerTable   = cms.untracked.int32(printTriggers),
                                       minVertexNdf        = cms.untracked.double(10.),
                                       minVertexMeanWeight = cms.untracked.double(0.5),
                                       profilePtBorders = cms.untracked.vdouble(ptBorders),
                                       iovList = cms.untracked.vint32(iovListList)
                                       )


process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("{}/JetHTAnalysis.root".format(configuration.get("output", os.getcwd()))),	
                                   closeFileFast = cms.untracked.bool(False)
                                   )


if (triggerFilter == "nothing" or triggerFilter == ""):
    process.p = cms.Path(process.offlineBeamSpot                        + 
                         process.TrackRefitter                          + 
                         process.offlinePrimaryVerticesFromRefittedTrks +
                         process.jetHTAnalyzer)
else:
    process.p = cms.Path(process.HLTFilter                              +
                         process.offlineBeamSpot                        + 
                         process.TrackRefitter                          + 
                         process.offlinePrimaryVerticesFromRefittedTrks +
                         process.jetHTAnalyzer)


