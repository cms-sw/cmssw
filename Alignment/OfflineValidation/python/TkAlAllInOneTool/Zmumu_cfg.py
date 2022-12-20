import math 
import json
import os

import FWCore.ParameterSet.Config as cms
import FWCore.PythonUtilities.LumiList as LumiList
from FWCore.ParameterSet.VarParsing import VarParsing
from Alignment.OfflineValidation.TkAlAllInOneTool.utils import _byteify

###################################################################
# Define process
###################################################################
process = cms.Process("TkAlignmentDiMuonValidation")

###################################################################
# Argument parsing
###################################################################
options = VarParsing()
options.register("config", "", VarParsing.multiplicity.singleton, VarParsing.varType.string , "AllInOne config")
options.parseArguments()

##Set validation mode
valiMode = "StandAlone"

###################################################################
# Read in AllInOne config in JSON format
###################################################################
with open(options.config, "r") as configFile:
    config = _byteify(json.load(configFile, object_hook=_byteify),ignore_dicts=True)

###################################################################
# Read filenames from given TXT file
###################################################################
readFiles = []

with open(config["validation"]["dataset"], "r") as datafiles:
    for fileName in datafiles.readlines():
        readFiles.append(fileName.replace("\n", ""))

###################################################################
# Get good lumi section
###################################################################
if "goodlumi" in config["validation"]:
    if os.path.isfile(config["validation"]["goodlumi"]):
        goodLumiSecs = cms.untracked.VLuminosityBlockRange(LumiList.LumiList(filename = config["validation"]["goodlumi"]).getCMSSWString().split(','))

    else:
        print("Does not exist: {}. Continue without good lumi section file.")
        goodLumiSecs = cms.untracked.VLuminosityBlockRange()

else:
    goodLumiSecs = cms.untracked.VLuminosityBlockRange()

###################################################################
# Define input source
###################################################################
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(readFiles),
                            lumisToProcess = goodLumiSecs,
                            skipEvents = cms.untracked.uint32(0)
                        )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(config["validation"].get("maxevents", 2000000))
)

###################################################################
# Bookeeping
###################################################################
process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"),
   fileMode  =  cms.untracked.string('NOMERGE'),
)

###################################################################
# Standard includes
###################################################################
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

###################################################################
# Messages
###################################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

###################################################################
# Load and configure Track Refitter
###################################################################
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter = process.TrackRefitterP5.clone(
    src =  'ALCARECOTkAlZMuMu',
    TrajectoryInEvent = True,
    TTRHBuilder = "WithAngleAndTemplate",
    NavigationSchool = "",
)

###################################################################
# Global Tag
###################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, config["alignment"].get("globaltag", "auto:run2_data"))

###################################################################
# Load conditions if wished
###################################################################
import CalibTracker.Configuration.Common.PoolDBESSource_cfi  
if "conditions" in config["alignment"]:
    from CalibTracker.Configuration.Common.PoolDBESSource_cfi import poolDBESSource
    for condition in config["alignment"]["conditions"]:
        setattr(process, "conditionsIn{}".format(condition), poolDBESSource.clone(
            connect = cms.string(str(config["alignment"]["conditions"][condition]["connect"])),
            toGet = cms.VPSet(cms.PSet(record = cms.string(str(condition)),
                                       tag = cms.string(str(config["alignment"]["conditions"][condition]["tag"]))
                                      )
                              )
            )
        )

        setattr(process, "prefer_conditionsIn{}".format(condition), cms.ESPrefer("PoolDBESSource", "conditionsIn{}".format(condition)))

###################################################################
# The Di Muon Mass Validation module
###################################################################
process.DiMuonMassValidation = cms.EDAnalyzer('DiMuonValidation',
                                              TkTag = cms.string ('TrackRefitter'),
                                              # mu mu mass
                                              Pair_mass_min = cms.double(80.),
                                              Pair_mass_max = cms.double(120.),
                                              Pair_mass_nbins = cms.int32(80),
                                              Pair_etaminpos = cms.double(-1),
                                              Pair_etamaxpos = cms.double(1),
                                              Pair_etaminneg = cms.double(-1),
                                              Pair_etamaxneg = cms.double(1),

                                              # cosTheta CS
                                              Variable_CosThetaCS_xmin = cms.double(-1.),
                                              Variable_CosThetaCS_xmax = cms.double(1.),
                                              Variable_CosThetaCS_nbins = cms.int32(20),
                                              # DeltaEta
                                              Variable_DeltaEta_xmin = cms.double(-4.8),
                                              Variable_DeltaEta_xmax = cms.double(4.8),
                                              Variable_DeltaEta_nbins = cms.int32(20),
                                              # EtaMinus
                                              Variable_EtaMinus_xmin = cms.double(-2.4),
                                              Variable_EtaMinus_xmax = cms.double(2.4),
                                              Variable_EtaMinus_nbins = cms.int32(12),
                                              # EtaPlus
                                              Variable_EtaPlus_xmin = cms.double(-2.4),
                                              Variable_EtaPlus_xmax = cms.double(2.4),
                                              Variable_EtaPlus_nbins = cms.int32(12),
                                              # Phi CS
                                              Variable_PhiCS_xmin = cms.double(-math.pi/2.),
                                              Variable_PhiCS_xmax = cms.double(math.pi/2.),
                                              Variable_PhiCS_nbins = cms.int32(20),
                                              # Phi Minus
                                              Variable_PhiMinus_xmin = cms.double(-math.pi),
                                              Variable_PhiMinus_xmax = cms.double(math.pi),
                                              Variable_PhiMinus_nbins = cms.int32(16),
                                              # Phi Plus
                                              Variable_PhiPlus_xmin = cms.double(-math.pi),
                                              Variable_PhiPlus_xmax = cms.double(math.pi),
                                              Variable_PhiPlus_nbins = cms.int32(16),
                                              # mu mu pT
                                              Variable_PairPt_xmin = cms.double(0.),
                                              Variable_PairPt_xmax = cms.double(100.),
                                              Variable_PairPt_nbins= cms.int32(100)),
                                              #


###################################################################
# Define sequences depending on validation mode
###################################################################
if valiMode == "StandAlone":
    # Output file
    process.TFileService = cms.Service("TFileService",
            fileName = cms.string("{}/Zmumu.root".format(config["output"])),
            closeFileFast = cms.untracked.bool(True),
    )

process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter*process.DiMuonMassValidation)
