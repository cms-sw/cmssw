import os
import FWCore.ParameterSet.Config as cms

inputfiles = os.environ["ALIGNMENT_INPUTFILES"].split(" ")
inputdb = os.environ["ALIGNMENT_INPUTDB"]
iteration = int(os.environ["ALIGNMENT_ITERATION"])
DIRNAME = os.environ["ALIGNMENT_DIRNAME"]
mode = os.environ["ALIGNMENT_MODE"]
params = os.environ["ALIGNMENT_PARAMS"]
minhits = int(os.environ["ALIGNMENT_MINHITS"])
mintracks = int(os.environ["ALIGNMENT_MINTRACKS"])
combineME11 = (os.environ["ALIGNMENT_COMBINEME11"] == "True")

process = cms.Process("ALIGN")
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(*inputfiles))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring("cout"),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string("ERROR")))

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")
process.looper.doTracker = cms.untracked.bool(False)
process.looper.doMuon = cms.untracked.bool(True)
process.looper.tjTkAssociationMapTag = cms.InputTag("CSCOverlapsTrackPreparation")
process.looper.algoConfig = cms.PSet(
    algoName = cms.string("CSCOverlapsAlignmentAlgorithm"),
    mode = cms.string(mode),
    maxHitErr = cms.double(0.2),        # exclude the last strip
    minHitsPerChamber = cms.int32(minhits),
    beamlineAngle = cms.double(0.10),   # don't accept segments that fail to point at the beamline
    maxRotYDiff = cms.double(0.030),
    maxRPhiDiff = cms.double(1.5),
    maxRedChi2 = cms.double(10.),
    minTracksPerAlignable = cms.int32(mintracks),
    useHitWeightsInTrackFit = cms.bool(True),
    useFitWeightsInMean = cms.bool(False),
    makeHistograms = cms.bool(True),
    combineME11 = cms.bool(combineME11),
    )

process.looper.ParameterBuilder.Selector.alignParams = cms.vstring("MuonCSCChambers,%s" % params)

process.CSCOverlapsBeamSplashCut = cms.EDFilter("CSCOverlapsBeamSplashCut", src = cms.InputTag("cscSegments"), maxSegments = cms.int32(30))
process.CSCOverlapsTrackPreparation = cms.EDProducer("CSCOverlapsTrackPreparation", src = cms.InputTag("ALCARECOMuAlBeamHaloOverlaps"))
process.Path = cms.Path(process.offlineBeamSpot * process.CSCOverlapsBeamSplashCut * process.CSCOverlapsTrackPreparation)

import CondCore.DBCommon.CondDBSetup_cfi
process.inertGlobalPositionRcd = cms.ESSource("PoolDBESSource",
                                              CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
                                              connect = cms.string("sqlite_file:inertGlobalPositionRcd.db"),
                                              toGet = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string("inertGlobalPositionRcd"))))
process.fakeTrackerAlignment = cms.ESSource("PoolDBESSource",
                                            CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
                                            connect = cms.string("frontier://FrontierProd/CMS_COND_31X_FROM21X"),
                                            toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),       tag = cms.string("TrackerIdealGeometry210_mc")),
                                                              cms.PSet(record = cms.string("TrackerAlignmentErrorRcd"),  tag = cms.string("TrackerIdealGeometryErrors210_mc"))))
process.muonAlignment = cms.ESSource("PoolDBESSource",
                                     CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
                                     connect = cms.string("sqlite_file:%s" % inputdb),
                                     toGet = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"),       tag = cms.string("DTAlignmentRcd")),
                                                       cms.PSet(record = cms.string("DTAlignmentErrorRcd"),  tag = cms.string("DTAlignmentErrorRcd")),
                                                       cms.PSet(record = cms.string("CSCAlignmentRcd"),      tag = cms.string("CSCAlignmentRcd")),
                                                       cms.PSet(record = cms.string("CSCAlignmentErrorRcd"), tag = cms.string("CSCAlignmentErrorRcd"))))
process.looper.applyDbAlignment = True

process.looper.saveToDB = True
process.looper.saveApeToDB = True
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBSetup,
                                          connect = cms.string("sqlite_file:%s_%02d.db" % (DIRNAME, iteration)),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
                                                            cms.PSet(record = cms.string("DTAlignmentErrorRcd"), tag = cms.string("DTAlignmentErrorRcd")),
                                                            cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
                                                            cms.PSet(record = cms.string("CSCAlignmentErrorRcd"), tag = cms.string("CSCAlignmentErrorRcd"))))

process.TFileService = cms.Service("TFileService", fileName = cms.string("%s_%02d.root" % (DIRNAME, iteration)))
