import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from Configuration.StandardSequences.GeometryExtended_cff import *
from Configuration.StandardSequences.MagneticField_cff import *
from TrackingTools.GeomPropagators.SmartPropagator_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
from RecoTracker.Configuration.RecoTracker_cff import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
from TrackingTools.TrackRefitter.TracksToTrajectories_cff import *
TrackerDigiGeometryESModule.applyAlignment = False
del DTGeometryESModule
del CSCGeometryESModule

from Alignment.CommonAlignmentProducer.AlignmentProducer_cff import *
looper.doTracker = cms.untracked.bool(False)
looper.doMuon = cms.untracked.bool(True)
looper.ParameterBuilder.Selector.alignParams = cms.vstring("MuonCSCChambers,111111")
looper.tjTkAssociationMapTag = cms.InputTag("CSCOverlapsTrackPreparationTrackRefitter")

from Alignment.MuonAlignmentAlgorithms.CSCOverlapsAlignmentAlgorithm_cfi import *
looper.algoConfig = CSCOverlapsAlignmentAlgorithm

CSCOverlapsTrackPreparationTrackRefitter = cms.EDProducer("CSCOverlapsTrackPreparation", src = cms.InputTag("ALCARECOMuAlBeamHaloOverlaps"))
Path = cms.Path(offlineBeamSpot * CSCOverlapsTrackPreparationTrackRefitter)

import CondCore.DBCommon.CondDBSetup_cfi
inertGlobalPositionRcd = cms.ESSource("PoolDBESSource",
                                      CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
                                      connect = cms.string("sqlite_file:inertGlobalPositionRcd.db"),
                                      toGet = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string("inertGlobalPositionRcd"))))
es_prefer_inertGlobalPositionRcd = cms.ESPrefer("PoolDBESSource", "inertGlobalPositionRcd")
muonAlignment = cms.ESSource("PoolDBESSource",
                             CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
                             connect = cms.string("sqlite_file:inputdb.db"),
                             toGet = cms.VPSet(cms.PSet(record = cms.string("CSCAlignmentRcd"),      tag = cms.string("CSCAlignmentRcd"))))
es_prefer_muonAlignment = cms.ESPrefer("PoolDBESSource", "muonAlignment")
looper.applyDbAlignment = True

PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
                                  connect = cms.string("sqlite_file:outputdb.db"),
                                  toPut = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
                                                    cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"), tag = cms.string("DTAlignmentErrorExtendedRcd")),
                                                    cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
                                                    cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"), tag = cms.string("CSCAlignmentErrorExtendedRcd"))))
looper.saveToDB = True
looper.saveApeToDB = True
