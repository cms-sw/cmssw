import glob
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()

###################################################################
# Setup 'standard' options
###################################################################

options.register('OutFileName',
                 "test.root", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "name of the output file (test.root is default)")

options.register('myGT',
                 "auto:run2_data", # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string, # string, int, or float
                 "name of the input Global Tag")

options.register('maxEvents',
                 -1,
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list                 
                 VarParsing.VarParsing.varType.int, # string, int, or float
                 "num. events to run")

options.parseArguments()

process = cms.Process("AlCaRECOAnalysis")

###################################################################
# Message logger service
###################################################################
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

###################################################################
# Geometry producer and standard includes
###################################################################
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")
process.load("CondCore.CondDB.CondDB_cfi")

####################################################################
# Get the GlogalTag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag,options.myGT, '')

###################################################################
# Source
###################################################################
readFiles = cms.untracked.vstring()
readFiles.extend(['file:SiPixelCalSingleMuonTight.root'])
#readFiles.extend(['file:SiPixelCalSingleMuonTight_fullDetId.root'])
process.source = cms.Source("PoolSource",fileNames = readFiles)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))

###################################################################
# The TrackRefitter
###################################################################
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.TrackRefitter = process.TrackRefitterP5.clone(src = 'ALCARECOSiPixelCalSingleMuonTight',
                                                      TrajectoryInEvent = True,
                                                      TTRHBuilder = "WithAngleAndTemplate",
                                                      NavigationSchool = "",
                                                      )

###################################################################
# The analysis module
###################################################################
process.myanalysis = cms.EDAnalyzer("NearbyPixelClustersAnalyzer",
                                    trajectoryInput = cms.InputTag("TrackRefitter"),
                                    #skimmedGeometryPath = cms.string("CalibTracker/SiPixelESProducers/data/PixelSkimmedGeometry.txt")  # phase-0
                                    skimmedGeometryPath = cms.string("SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt") #phase-1
                                    )

###################################################################
# Output name
###################################################################
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(options.OutFileName))

###################################################################
# Path
###################################################################
process.p1 = cms.Path(process.offlineBeamSpot*
                      process.TrackRefitter*
                      process.myanalysis)
