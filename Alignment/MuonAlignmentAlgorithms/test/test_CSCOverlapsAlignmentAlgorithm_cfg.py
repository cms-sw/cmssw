import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("ALIGN")
# process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:///NOBACKUP/results/MuAlBeamHaloOverlaps_MC1.root", "file:///NOBACKUP/results/MuAlBeamHaloOverlaps_MC2.root"))
# process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(30000))

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(0))

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring("cout"),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string("ERROR")))

process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load("Configuration/StandardSequences/GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("TrackingTools.GeomPropagators.SmartPropagator_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.load("RecoTracker.Configuration.RecoTracker_cff")
process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi")
process.load("TrackingTools.TrackRefitter.TracksToTrajectories_cff")
process.TrackerDigiGeometryESModule.applyAlignment = False
del process.DTGeometryESModule
del process.CSCGeometryESModule

process.TFileService = cms.Service("TFileService", fileName = cms.string("test2.root"))

process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")
process.looper.doTracker = cms.untracked.bool(False)
process.looper.doMuon = cms.untracked.bool(True)
process.looper.tjTkAssociationMapTag = cms.InputTag("CSCOverlapsTrackPreparationTrackRefitter")
process.looper.algoConfig = cms.PSet(
    algoName = cms.string("CSCOverlapsAlignmentAlgorithm"),

    TrackTransformer = cms.PSet(DoPredictionsOnly = cms.bool(False),
                                Fitter = cms.string("KFFitterForRefitInsideOut"),
                                TrackerRecHitBuilder = cms.string("WithoutRefit"),
                                Smoother = cms.string("KFSmootherForRefitInsideOut"),
                                MuonRecHitBuilder = cms.string("MuonRecHitBuilder"),
                                RefitDirection = cms.string("alongMomentum"),
                                RefitRPCHits = cms.bool(False),
                                Propagator = cms.string("SteppingHelixPropagatorAny")),

    mode = cms.string("phipos"),
    reportFileName = cms.string("reports.py"),
    minP = cms.double(5.),
    minHitsPerChamber = cms.int32(5),
    maxRedChi2 = cms.double(10.),
    fiducial = cms.bool(True),
    useHitWeights = cms.bool(True),
    slopeFromTrackRefit = cms.bool(False),
    minStationsInTrackRefits = cms.int32(2),
    combineME11 = cms.bool(True),
    useTrackWeights = cms.bool(False),
    errorFromRMS = cms.bool(False),
    writeTemporaryFile = cms.string(""),
    readTemporaryFiles = cms.vstring("test.tmp"),
    doAlignment = cms.bool(True),
    makeHistograms = cms.bool(True),

    fitters = cms.VPSet(
    cms.PSet(name = cms.string("ME+1/1"),
             alignables = cms.vstring("PGFrame",
                                      "ME+1/1/01", "ME+1/1/02", "ME+1/1/03", "ME+1/1/04", "ME+1/1/05", "ME+1/1/06", "ME+1/1/07", "ME+1/1/08", "ME+1/1/09", "ME+1/1/10", "ME+1/1/11", "ME+1/1/12", "ME+1/1/13", "ME+1/1/14", "ME+1/1/15", "ME+1/1/16", "ME+1/1/17", "ME+1/1/18", "ME+1/1/19", "ME+1/1/20", "ME+1/1/21", "ME+1/1/22", "ME+1/1/23", "ME+1/1/24", "ME+1/1/25", "ME+1/1/26", "ME+1/1/27", "ME+1/1/28", "ME+1/1/29", "ME+1/1/30", "ME+1/1/31", "ME+1/1/32", "ME+1/1/33", "ME+1/1/34", "ME+1/1/35", "ME+1/1/36"),
             fixed = cms.string(""),
             constraints = cms.VPSet(
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/01"), value = cms.double(0.005), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/02"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/03"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/04"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/05"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/06"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/07"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/08"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/09"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/10"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/11"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/12"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/13"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/14"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/15"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/16"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/17"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/18"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/19"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/20"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/21"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/22"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/23"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/24"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/25"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/26"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/27"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/28"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/29"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/30"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/31"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/32"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/33"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/34"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/35"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME+1/1/36"), value = cms.double(0.), error = cms.double(0.1)),
    
    )),

    cms.PSet(name = cms.string("ME-1/1"),
             alignables = cms.vstring("PGFrame",
                                      "ME-1/1/01", "ME-1/1/02", "ME-1/1/03", "ME-1/1/04", "ME-1/1/05", "ME-1/1/06", "ME-1/1/07", "ME-1/1/08", "ME-1/1/09", "ME-1/1/10", "ME-1/1/11", "ME-1/1/12", "ME-1/1/13", "ME-1/1/14", "ME-1/1/15", "ME-1/1/16", "ME-1/1/17", "ME-1/1/18", "ME-1/1/19", "ME-1/1/20", "ME-1/1/21", "ME-1/1/22", "ME-1/1/23", "ME-1/1/24", "ME-1/1/25", "ME-1/1/26", "ME-1/1/27", "ME-1/1/28", "ME-1/1/29", "ME-1/1/30", "ME-1/1/31", "ME-1/1/32", "ME-1/1/33", "ME-1/1/34", "ME-1/1/35", "ME-1/1/36"),
             fixed = cms.string(""),
             constraints = cms.VPSet(
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/01"), value = cms.double(0.005), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/02"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/03"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/04"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/05"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/06"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/07"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/08"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/09"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/10"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/11"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/12"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/13"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/14"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/15"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/16"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/17"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/18"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/19"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/20"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/21"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/22"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/23"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/24"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/25"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/26"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/27"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/28"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/29"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/30"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/31"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/32"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/33"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/34"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/35"), value = cms.double(0.), error = cms.double(0.1)),
    cms.PSet(i = cms.string("PGFrame"), j = cms.string("ME-1/1/36"), value = cms.double(0.), error = cms.double(0.1)),
    
    )),

    ))

#     cms.PSet(i = cms.string("SLM1Frame"), j = cms.string("ME+2/1/02"), value = cms.double(0.), error = cms.double(1.)),
#     cms.PSet(i = cms.string("SLM1Frame"), j = cms.string("ME+2/1/10"), value = cms.double(0.), error = cms.double(1.)),
#     cms.PSet(i = cms.string("SLM2Frame"), j = cms.string("ME+2/1/04"), value = cms.double(0.), error = cms.double(1.)),
#     cms.PSet(i = cms.string("SLM2Frame"), j = cms.string("ME+2/1/14"), value = cms.double(0.), error = cms.double(1.)),
#     cms.PSet(i = cms.string("SLM3Frame"), j = cms.string("ME+2/1/08"), value = cms.double(0.), error = cms.double(1.)),
#     cms.PSet(i = cms.string("SLM3Frame"), j = cms.string("ME+2/1/16"), value = cms.double(0.), error = cms.double(1.)),

process.looper.ParameterBuilder.Selector.alignParams = cms.vstring("MuonCSCChambers,111111")

process.CSCOverlapsTrackPreparationTrackRefitter = cms.EDProducer("CSCOverlapsTrackPreparation", src = cms.InputTag("ALCARECOMuAlBeamHaloOverlaps"))
# process.TrackRefitter = cms.EDProducer("TracksToTrajectories",
#                                        Type = cms.string("Default"),
#                                        Tracks = cms.InputTag("ALCARECOMuAlBeamHaloOverlaps"),
#                                        TrackTransformer = cms.PSet(DoPredictionsOnly = cms.bool(False),
#                                                                    Fitter = cms.string("KFFitterForRefitInsideOut"),
#                                                                    TrackerRecHitBuilder = cms.string("WithoutRefit"),
#                                                                    Smoother = cms.string("KFSmootherForRefitInsideOut"),
#                                                                    MuonRecHitBuilder = cms.string("MuonRecHitBuilder"),
#                                                                    RefitDirection = cms.string("alongMomentum"),
#                                                                    RefitRPCHits = cms.bool(False),
#                                                                    Propagator = cms.string("SmartPropagatorAnyRKOpposite")))
# process.TrackRefitter = cms.EDProducer("TracksToTrajectories",
#                                        Type = cms.string("CosmicMuonsForAlignment"),
#                                        Tracks = cms.InputTag("ALCARECOMuAlBeamHaloOverlaps"),
#                                        TrackTransformer = cms.PSet(TrackerRecHitBuilder = cms.string("WithoutRefit"),
#                                                                    MuonRecHitBuilder = cms.string("MuonRecHitBuilder"),
#                                                                    RefitRPCHits = cms.bool(False)))
process.Path = cms.Path(process.offlineBeamSpot * process.CSCOverlapsTrackPreparationTrackRefitter)

import CondCore.DBCommon.CondDBSetup_cfi
process.inertGlobalPositionRcd = cms.ESSource("PoolDBESSource",
                                              CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
                                              connect = cms.string("sqlite_file:inertGlobalPositionRcd.db"),
                                              toGet = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string("inertGlobalPositionRcd"))))
process.muonAlignment = cms.ESSource("PoolDBESSource",
                                     CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
                                     connect = cms.string("sqlite_file:geometry.db"),   # ideal.db, Photogrammetry.db
                                     toGet = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"),       tag = cms.string("DTAlignmentRcd")),
                                                       cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"),  tag = cms.string("DTAlignmentErrorExtendedRcd")),
                                                       cms.PSet(record = cms.string("CSCAlignmentRcd"),      tag = cms.string("CSCAlignmentRcd")),
                                                       cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"), tag = cms.string("CSCAlignmentErrorExtendedRcd"))))
process.looper.applyDbAlignment = True

process.looper.saveToDB = True
process.looper.saveApeToDB = True
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBSetup,
                                          connect = cms.string("sqlite_file:after.db"),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
                                                            cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"), tag = cms.string("DTAlignmentErrorExtendedRcd")),
                                                            cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
                                                            cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"), tag = cms.string("CSCAlignmentErrorExtendedRcd"))))
