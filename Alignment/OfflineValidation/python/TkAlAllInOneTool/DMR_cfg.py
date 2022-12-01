import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
from Alignment.OfflineValidation.TkAlAllInOneTool.defaultInputFiles_cff import filesDefaultMC_NoPU

from FWCore.ParameterSet.VarParsing import VarParsing

import json
import os

##Define process
process = cms.Process("OfflineValidator")

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

print(config)

##Read filenames from given TXT file and define input source
readFiles = []

if "dataset" in config["validation"]:
    with open(config["validation"]["dataset"], "r") as datafiles:
        for fileName in datafiles.readlines():
            readFiles.append(fileName.replace("\n", ""))

    ##Define input source
    process.source = cms.Source("PoolSource",
                                fileNames = cms.untracked.vstring(readFiles),
                                skipEvents = cms.untracked.uint32(0)
                            )
else:
    print(">>>>>>>>>> DMR_cfg.py: msg%-i: config not specified! Loading default MC simulation -> filesDefaultMC_NoPU!")
    process.source = cms.Source("PoolSource",
                                fileNames = filesDefaultMC_NoPU,
                                skipEvents = cms.untracked.uint32(0)
                            )

##Get good lumi section
if "goodlumi" in config["validation"]:
    if os.path.isfile(config["validation"]["goodlumi"]):
        goodLumiSecs = cms.untracked.VLuminosityBlockRange(LumiList.LumiList(filename = config["validation"]["goodlumi"]).getCMSSWString().split(','))

    else:
        print("Does not exist: {}. Continue without good lumi section file.")
        goodLumiSecs = cms.untracked.VLuminosityBlockRange()

else:
    goodLumiSecs = cms.untracked.VLuminosityBlockRange()

process.source.lumisToProcess = goodLumiSecs

##default set to 1 for unit tests
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(config["validation"].get("maxevents", 1))
)

##Bookeeping
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"),
   fileMode  =  cms.untracked.string('NOMERGE'),
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
       destinations   = cms.untracked.vstring('cerr'),
       cerr       = cms.untracked.PSet(
                    threshold = cms.untracked.string('ERROR')
        )

)

##Basic modules
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.Geometry.GeometryDB_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")

##Track fitting
import Alignment.CommonAlignment.tools.trackselectionRefitting as trackselRefit
process.seqTrackselRefit = trackselRefit.getSequence(process,
                                                     config["validation"].get("trackcollection", "generalTracks"),
                                                     isPVValidation = False, 
                                                     TTRHBuilder = config["validation"].get("tthrbuilder", "WithAngleAndTemplate"),
                                                     usePixelQualityFlag=config["validation"].get("usePixelQualityFlag", True),
                                                     openMassWindow = False,
                                                     cosmicsDecoMode = True,
                                                     cosmicsZeroTesla=config["validation"].get("cosmicsZeroTesla", False),
                                                     momentumConstraint = None,
                                                     cosmicTrackSplitting = False,
                                                     use_d0cut = True,
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

##Filter good events
process.oneGoodVertexFilter = cms.EDFilter("VertexSelector",
                                           src = cms.InputTag(config["validation"].get("vertexcollection", "offlinePrimaryVertices")),
                                           cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"),
                                           filter = cms.bool(True),
)
process.FilterGoodEvents=cms.Sequence(process.oneGoodVertexFilter)

process.noScraping= cms.EDFilter("FilterOutScraping",
                                 src=cms.untracked.InputTag(config["validation"].get("trackcollection", "generalTracks")),
                                 applyfilter = cms.untracked.bool(True),
                                 debugOn = cms.untracked.bool(False),
                                 numtrack = cms.untracked.uint32(10),
                                 thresh = cms.untracked.double(0.25),
)

##Offline validation analyzer
process.TrackerOfflineValidation = cms.EDAnalyzer("TrackerOfflineValidation",
    useInDqmMode              = cms.bool(False), 
    moduleDirectoryInOutput   = cms.string(""), 
    Tracks                    = cms.InputTag("FinalTrackRefitter"),
    trajectoryInput           = cms.string('FinalTrackRefitter'),  # Only needed in DQM mode
    localCoorHistosOn         = cms.bool(False),
    moduleLevelHistsTransient = config["validation"].get("moduleLevelHistsTransient", cms.bool(False)), 
    moduleLevelProfiles       = config["validation"].get("moduleLevelProfiles", cms.bool(True)),
    localCoorProfilesOn       = cms.bool(False),
    stripYResiduals           = cms.bool(config["validation"].get("stripYResiduals", False)),
    useFwhm                   = cms.bool(True),
    useFit                    = cms.bool(False),
    useCombinedTrajectory     = cms.bool(False),
    useOverflowForRMS         = cms.bool(False),
    maxTracks                 = cms.uint64(config["validation"].get("maxtracks", 1)),
    chargeCut                 = cms.int32(config["validation"].get("chargecut", 0)),
    # Normalized X Residuals, normal local coordinates (Strip)
    TH1NormXResStripModules = cms.PSet(
        Nbinx = cms.int32(100), xmin = cms.double(-5.0), xmax = cms.double(5.0)
    ),

    # X Residuals, normal local coordinates (Strip)
    TH1XResStripModules = cms.PSet(
        Nbinx = cms.int32(100), xmin = cms.double(-0.5), xmax = cms.double(0.5)
    ),

    # Normalized X Residuals, native coordinates (Strip)
    TH1NormXprimeResStripModules = cms.PSet(
        Nbinx = cms.int32(120), xmin = cms.double(-3.0), xmax = cms.double(3.0)
    ),

    # X Residuals, native coordinates (Strip)
    TH1XprimeResStripModules = cms.PSet(
        Nbinx = cms.int32(5000), xmin = cms.double(-0.05), xmax = cms.double(0.05)
    ),

    # Normalized Y Residuals, native coordinates (Strip -> hardly defined)
    TH1NormYResStripModules = cms.PSet(
        Nbinx = cms.int32(120), xmin = cms.double(-3.0), xmax = cms.double(3.0)
    ),
    # -> very broad distributions expected
    TH1YResStripModules = cms.PSet(
        Nbinx = cms.int32(5000), xmin = cms.double(-11.0), xmax = cms.double(11.0)
    ),

    # Normalized X residuals normal local coordinates (Pixel)
    TH1NormXResPixelModules = cms.PSet(
        Nbinx = cms.int32(100), xmin = cms.double(-5.0), xmax = cms.double(5.0)
    ),
    # X residuals normal local coordinates (Pixel)
    TH1XResPixelModules = cms.PSet(
        Nbinx = cms.int32(100), xmin = cms.double(-0.5), xmax = cms.double(0.5)
    ),
    # Normalized X residuals native coordinates (Pixel)
    TH1NormXprimeResPixelModules = cms.PSet(
        Nbinx = cms.int32(120), xmin = cms.double(-3.0), xmax = cms.double(3.0)
    ),
    # X residuals native coordinates (Pixel)
    TH1XprimeResPixelModules = cms.PSet(
        Nbinx = cms.int32(5000), xmin = cms.double(-0.05), xmax = cms.double(0.05)
    ),
    # Normalized Y residuals native coordinates (Pixel)
    TH1NormYResPixelModules = cms.PSet(
        Nbinx = cms.int32(120), xmin = cms.double(-3.0), xmax = cms.double(3.0)
    ),
    # Y residuals native coordinates (Pixel)
    TH1YResPixelModules = cms.PSet(
        Nbinx = cms.int32(5000), xmin = cms.double(-0.05), xmax = cms.double(0.05)
    ),
    # X Residuals vs reduced local coordinates (Strip)
    TProfileXResStripModules = cms.PSet(
        Nbinx = cms.int32(34), xmin = cms.double(-1.02), xmax = cms.double(1.02)
    ),
    # X Residuals vs reduced local coordinates (Strip)
    TProfileYResStripModules = cms.PSet(
        Nbinx = cms.int32(34), xmin = cms.double(-1.02), xmax = cms.double(1.02)
    ),
    # X Residuals vs reduced local coordinates (Pixel)
    TProfileXResPixelModules = cms.PSet(
        Nbinx = cms.int32(17), xmin = cms.double(-1.02), xmax = cms.double(1.02)
    ),
    # X Residuals vs reduced local coordinates (Pixel)
    TProfileYResPixelModules = cms.PSet(
        Nbinx = cms.int32(17), xmin = cms.double(-1.02), xmax = cms.double(1.02)
    ),
)

##Output file
process.TFileService = cms.Service("TFileService",
        fileName = cms.string("{}/DMR.root".format(config.get("output", os.getcwd()))),
        closeFileFast = cms.untracked.bool(True),
)

seqTrackerOfflineValidation = cms.Sequence(process.TrackerOfflineValidation)

##Let all sequences run
process.p = cms.Path(process.seqTrackselRefit*seqTrackerOfflineValidation)
