import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultData_MinBias2018B

from FWCore.ParameterSet.VarParsing import VarParsing

from Alignment.OfflineValidation.TkAlAllInOneTool.utils import _byteify
import json
import os

##Define process
process = cms.Process("PrimaryVertexValidation")

##Argument parsing
options = VarParsing()
options.register("config", "", VarParsing.multiplicity.singleton, VarParsing.varType.string , "AllInOne config")

options.parseArguments()

##Read in AllInOne config in JSON format
if options.config == "":
    config = {"validation": {},
              "alignment": {}}
else:
    with open(options.config, "r") as configFile:
        config = json.load(configFile)

isDA = config["validation"].get("isda", True)
isMC = config["validation"].get("ismc", True)

runboundary = config["validation"].get("runboundary", 1)

##Read filenames from given TXT file and define input source
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
    print(">>>>>>>>>> PV_cfg.py: msg%-i: config not specified! Loading default dataset -> filesDefaultData_MinBias2018B!")
    process.source = cms.Source("PoolSource",
                                fileNames = filesDefaultData_MinBias2018B,
                                skipEvents = cms.untracked.uint32(0)
                            )

##Get good lumi section and load data or handle MC
if "goodlumi" in config["validation"]:
    if os.path.isfile(config["validation"]["goodlumi"]):
        goodLumiSecs = cms.untracked.VLuminosityBlockRange(LumiList.LumiList(filename = config["validation"]["goodlumi"]).getCMSSWString().split(','))
        
    else:
        print("Does not exist: {}. Continue without good lumi section file.")
        goodLumiSecs = cms.untracked.VLuminosityBlockRange()

else:
    goodLumiSecs = cms.untracked.VLuminosityBlockRange()

if isMC:
     print(">>>>>>>>>> PV_cfg.py: msg%-i: This is simulation!")
     runboundary = 1
else:
     process.source.lumisToProcess = goodLumiSecs

isMultipleRuns=False
if(isinstance(runboundary, (list, tuple))):
     isMultipleRuns=True
     print("Multiple Runs are selected")       
if(isMultipleRuns):
     process.source.firstRun = cms.untracked.uint32(runboundary[0])
else:
     process.source.firstRun = cms.untracked.uint32(runboundary)

##default set to 1 for unit tests
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(config["validation"].get("maxevents", 1)))

##Bookeeping
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"),
   fileMode  =  cms.untracked.string('NOMERGE'),
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations   = cms.untracked.vstring('cerr'),
                                    cerr       = cms.untracked.PSet(threshold = cms.untracked.string('INFO'))
                                   )

##Basic modules
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.Geometry.GeometryRecoDB_cff") #or process.load("Configuration.Geometry.GeometryDB_cff")?????
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")

####################################################################
# Produce the Transient Track Record in the event
####################################################################
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

####################################################################
# Load and Configure Common Track Selection and refitting sequence
####################################################################
import Alignment.CommonAlignment.tools.trackselectionRefitting as trackselRefit
process.seqTrackselRefit = trackselRefit.getSequence(process,
                                                     config["validation"].get("trackcollection", "ALCARECOTkAlMinBias"),
                                                     isPVValidation=True,
                                                     TTRHBuilder=config["validation"].get("tthrbuilder", "WithAngleAndTemplate"),
                                                     usePixelQualityFlag=config["validation"].get("usePixelQualityFlag", True),
                                                     openMassWindow=False,
                                                     cosmicsDecoMode=True,
                                                     cosmicsZeroTesla=config["validation"].get("cosmicsZeroTesla", False),                                                     
                                                     momentumConstraint=None,
                                                     cosmicTrackSplitting=False,
                                                     use_d0cut=False,
                                                     )

#Global tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, config["alignment"].get("globaltag", "auto:phase1_2017_realistic"))

##Load conditions if wished
if "conditions" in config["alignment"]:
    from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource

    for condition in config["alignment"]["conditions"]:
        setattr(process, "conditionsIn{}".format(condition), poolDBESSource.clone(
             connect = cms.string(str(config["alignment"]["conditions"][condition]["connect"])),
             toGet = cms.VPSet(
                        cms.PSet(
                                 record = cms.string(str(condition)),
                                 tag = cms.string(str(config["alignment"]["conditions"][condition]["tag"]))
                        )
                     )
            )
        )

        setattr(process, "prefer_conditionsIn{}".format(condition), cms.ESPrefer("PoolDBESSource", "conditionsIn{}".format(condition)))

####################################################################
# Load and Configure event selection
####################################################################
process.primaryVertexFilter = cms.EDFilter("VertexSelector",
                                           src = cms.InputTag(config["validation"].get("vertexcollection", "offlinePrimaryVertices")),
                                           cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),
                                           filter = cms.bool(True)
                                           )

process.noscraping = cms.EDFilter("FilterOutScraping",
                                  applyfilter = cms.untracked.bool(True),
                                  src = cms.untracked.InputTag(config["validation"].get("trackcollection", "ALCARECOTkAlMinBias")),
                                  debugOn = cms.untracked.bool(False),
                                  numtrack = cms.untracked.uint32(10),
                                  thresh = cms.untracked.double(0.25)
                                  )

process.load("Alignment.CommonAlignment.filterOutLowPt_cfi")
process.filterOutLowPt.src = cms.untracked.InputTag(config["validation"].get("trackcollection", "ALCARECOTkAlMinBias"))
process.filterOutLowPt.ptmin = cms.untracked.double(config["validation"].get("ptCut", 3.))
process.filterOutLowPt.runControl = False
if(isMultipleRuns):
     process.filterOutLowPt.runControlNumber.extend((runboundary))
else:
     process.filterOutLowPt.runControlNumber = [runboundary]

if isMC:
     process.goodvertexSkim = cms.Sequence(process.noscraping + process.filterOutLowPt)
else:
     process.goodvertexSkim = cms.Sequence(process.primaryVertexFilter + process.noscraping + process.filterOutLowPt)


####################################################################
# Imports of parameters
####################################################################
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices
## modify the parameters which differ
FilteringParams = offlinePrimaryVertices.TkFilterParameters.clone(
     maxNormalizedChi2 = 5.0,  # chi2ndof < 5
     maxD0Significance = 5.0,  # fake cut (requiring 1 PXB hit)
     maxEta = 5.0,             # as per recommendation in PR #18330
)

## MM 04.05.2017 (use settings as in: https://github.com/cms-sw/cmssw/pull/18330)
from RecoVertex.PrimaryVertexProducer.TkClusParameters_cff import DA_vectParameters
DAClusterizationParams = DA_vectParameters.clone()

GapClusterizationParams = cms.PSet(algorithm   = cms.string('gap'),
                                   TkGapClusParameters = cms.PSet(zSeparation = cms.double(0.2))  # 0.2 cm max separation betw. clusters
                                   )

####################################################################
# Deterministic annealing clustering or Gap clustering
####################################################################
def switchClusterizerParameters(da):
     if da:
          print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running DA Algorithm!")
          return DAClusterizationParams
     else:
          print(">>>>>>>>>> testPVValidation_cfg.py: msg%-i: Running GAP Algorithm!")
          return GapClusterizationParams

####################################################################
# Configure the PVValidation Analyzer module
####################################################################
from Alignment.OfflineValidation.primaryVertexValidation_cfi import primaryVertexValidation  as _primaryVertexValidation
process.PVValidation = _primaryVertexValidation.clone(
    TrackCollectionTag = "FinalTrackRefitter",
    VertexCollectionTag = config["validation"].get("vertexcollection", "offlinePrimaryVertices"),
    Debug = False,
    storeNtuple = False,
    useTracksFromRecoVtx = False,
    isLightNtuple = True,
    askFirstLayerHit = False,
    forceBeamSpot = config["validation"].get("forceBeamSpot", False),
    probePt = config["validation"].get("ptCut", 3),
    probeEta  = config["validation"].get("etaCut", 2.5),
    minPt  = config["validation"].get("minPt", 1.),
    maxPt  = config["validation"].get("maxPt", 30.),
    doBPix = config["validation"].get("doBPix", True),
    doFPix = config["validation"].get("doFPix", True),
    numberOfBins = config["validation"].get("numberOfBins", 48),
    runControl = config["validation"].get("runControl", False),
    runControlNumber = [runboundary],
    TkFilterParameters = FilteringParams,
    TkClusParameters = switchClusterizerParameters(isDA)
)

####################################################################
# Output file
####################################################################
process.TFileService = cms.Service("TFileService",
            fileName = cms.string("{}/PVValidation_{}_{}.root".format(config.get("output", os.getcwd()), config["alignment"].get("name", ""), config["validation"].get("IOV", 1.))),
            closeFileFast = cms.untracked.bool(True),
    )

####################################################################
# Path
####################################################################
process.p = cms.Path(process.goodvertexSkim*process.seqTrackselRefit*process.PVValidation)

print("Done")
