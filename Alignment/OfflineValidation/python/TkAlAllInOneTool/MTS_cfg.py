import json
import yaml
import os
import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultData_Cosmics_string
from FWCore.ParameterSet.VarParsing import VarParsing
from Alignment.OfflineValidation.TkAlAllInOneTool.utils import _byteify
import pdb

###################################################################
# Define process
###################################################################
process = cms.Process("splitter")

###################################################################
# Argument parsing
###################################################################
options = VarParsing()
options.register("config", "", VarParsing.multiplicity.singleton, VarParsing.varType.string , "AllInOne config")
options.parseArguments()

###################################################################
# Read in AllInOne config in JSON format
###################################################################
if options.config == "":
  config = {"validation": {},
            "alignment": {}}
else:
  with open(options.config, "r") as configFile:
    if options.config.endswith(".json"):
      config = json.load(configFile)
    elif options.config.endswith(".yaml"):
      config = yaml.safe_load(configFile)

###################################################################
# Read filenames from given TXT file and define input source
###################################################################
readFiles = []

if "dataset" in config["validation"]:
    with open(config["validation"]["dataset"], "r") as datafiles:
        for fileName in datafiles.readlines():
            readFiles.append(fileName.replace("\n", ""))

    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(readFiles),
                                skipEvents = cms.untracked.uint32(0)
                               )
else:
    print(">>>>>>>>>> MTS_cfg.py: msg%-i: dataset not specified! Loading default file -> filesDefaultData_Cosmics_string!")
    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(filesDefaultData_Cosmics_string),
                                skipEvents = cms.untracked.uint32(0)
                               )

###################################################################
# Get good lumi section and load data or handle MC
###################################################################
if "goodlumi" in config["validation"]:
     if os.path.isfile(config["validation"]["goodlumi"]):
          goodLumiSecs = cms.untracked.VLuminosityBlockRange(LumiList.LumiList(filename = config["validation"]["goodlumi"]).getCMSSWString().split(','))
     else:
          print("Does not exist: {}. Continue without good lumi section file.")
          goodLumiSecs = cms.untracked.VLuminosityBlockRange()

else:
     goodLumiSecs = cms.untracked.VLuminosityBlockRange()
process.source.lumisToProcess = goodLumiSecs

###################################################################
# Runs and events
###################################################################
runboundary = config["validation"].get("runboundary", 1)
isMultipleRuns=False
if(isinstance(runboundary, (list, tuple))):
     isMultipleRuns=True
     print("Multiple Runs are selected")
if(isMultipleRuns):
     process.source.firstRun = cms.untracked.uint32(runboundary[0])
else:
     process.source.firstRun = cms.untracked.uint32(runboundary)

###################################################################
# Default set to 1 for unit tests
###################################################################
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(config["validation"].get("maxevents", 1)))

###################################################################
# Bookeeping
###################################################################
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
   fileMode  =  cms.untracked.string('NOMERGE'), # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

###################################################################
# Messages
###################################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)

###################################################################
# Basic modules
###################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.Geometry.GeometryDB_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")

####################################################################
# Load and Configure Track refitter
####################################################################
import Alignment.CommonAlignment.tools.trackselectionRefitting as trackselRefit
process.seqTrackselRefit = trackselRefit.getSequence(
  process,
  config["validation"].get("trackcollection", "ALCARECOTkAlCosmicsCTF0T"),
  isPVValidation = False,
  TTRHBuilder = config["validation"].get("tthrbuilder", "WithAngleAndTemplate"),
  usePixelQualityFlag=config["validation"].get("usePixelQualityFlag", True),
  openMassWindow = False,
  cosmicsDecoMode = True,
  cosmicsZeroTesla=config["validation"].get("cosmicsZeroTesla", False),
  momentumConstraint = None,
  cosmicTrackSplitting = True,
  use_d0cut = False
)

####################################################################
# Global tag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, config["alignment"].get("globaltag", "124X_dataRun3_Prompt_v10"))

####################################################################
# Load conditions if wished
####################################################################
if "conditions" in config["alignment"]:
  from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource

  for condition in config["alignment"]["conditions"]:
    setattr(
      process,
      "conditionsIn{}".format(condition),
      poolDBESSource.clone(
        connect = cms.string(str(config["alignment"]["conditions"][condition]["connect"])),
        toGet = cms.VPSet(
          cms.PSet(
            record = cms.string(str(condition)),
            tag = cms.string(str(config["alignment"]["conditions"][condition]["tag"]))
          )
        )
      )
    )

    setattr(
      process,
      "prefer_conditionsIn{}".format(condition),
      cms.ESPrefer("PoolDBESSource", "conditionsIn{}".format(condition))
    )

####################################################################
# Configure the Analyzer module
####################################################################

process.FittingSmootherRKP5.EstimateCut = -1
process.AlignmentTrackSelector.minHitsPerSubDet.inPIXEL = 2
# Use compressions settings of TFile
# see https://root.cern.ch/root/html534/TFile.html#TFile:SetCompressionSet    tings
# settings = 100 * algorithm + level
# level is from 1 (small) to 9 (large compression)
# algo: 1 (ZLIB), 2 (LMZA)
# see more about compression & performance: https://root.cern.ch/root/html534/guides/users-guide/InputOutput.html#compression-and-performance
compressionSettings = 207
process.cosmicValidation = cms.EDAnalyzer(
  "CosmicSplitterValidation",
  compressionSettings = cms.untracked.int32(compressionSettings),
  ifSplitMuons = cms.bool(False),
  checkIfGolden = cms.bool(False),
  splitTracks = cms.InputTag("FinalTrackRefitter","","splitter"),
  splitGlobalMuons = cms.InputTag("muons","","splitter"),
  originalTracks = cms.InputTag("FirstTrackRefitter","","splitter"),
  originalGlobalMuons = cms.InputTag("muons","","Rec")
)

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
            fileName = cms.string("{}/MTSValidation_{}_{}.root".format(config.get("output", os.getcwd()), config["alignment"].get("name", ""), config["validation"].get("IOV", 1.))),
            closeFileFast = cms.untracked.bool(True),
    )

####################################################################
# Path
####################################################################
process.p = cms.Path(process.seqTrackselRefit*process.cosmicValidation)

print("Done")
-- dummy change --
