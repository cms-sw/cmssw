import FWCore.ParameterSet.Config as cms

### General track re-fitting includes
### (don't load dtGeometry_cfi or cscGeometry_cfi because it's provided by AlignmentProducer)
from Configuration.StandardSequences.MagneticField_cff import *
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
from Geometry.RPCGeometry.rpcGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *

### Track refitter for global collisions muons
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi import *
from TrackingTools.TrackRefitter.globalMuonTrajectories_cff import *
MuonAlignmentFromReferenceGlobalMuonRefit = globalMuons.clone()
MuonAlignmentFromReferenceGlobalMuonRefit.Tracks = cms.InputTag("ALCARECOMuAlCalIsolatedMu:GlobalMuon")
MuonAlignmentFromReferenceGlobalMuonRefit.RefitRPCHits = cms.bool(False)

### Track refitter for global cosmic muons
from TrackingTools.TrackRefitter.globalCosmicMuonTrajectories_cff import *
MuonAlignmentFromReferenceGlobalCosmicRefit = globalCosmicMuons.clone()
MuonAlignmentFromReferenceGlobalCosmicRefit.Tracks = cms.InputTag("ALCARECOMuAlGlobalCosmics:GlobalMuon")

### AlignmentProducer with basic options for muon alignment
from Alignment.CommonAlignmentProducer.AlignmentProducer_cff import *
looper.tjTkAssociationMapTag = cms.InputTag("MuonAlignmentFromReferenceGlobalMuonRefit:Refitted")
looper.doTracker = cms.untracked.bool(False)
looper.doMuon = cms.untracked.bool(True)
looper.ParameterBuilder.Selector = cms.PSet(
    alignParams = cms.vstring("MuonDTChambers,111111,stations123", "MuonDTChambers,100011,station4", "MuonCSCChambers,100011"),
    stations123 = cms.PSet(rRanges = cms.vdouble(0., 660.),
                           xRanges = cms.vdouble(), yRanges = cms.vdouble(), zRanges = cms.vdouble(), etaRanges = cms.vdouble(), phiRanges = cms.vdouble()),
    station4 = cms.PSet(rRanges = cms.vdouble(660., 800.),
                        xRanges = cms.vdouble(), yRanges = cms.vdouble(), zRanges = cms.vdouble(), etaRanges = cms.vdouble(), phiRanges = cms.vdouble()))

### MuonAlignmentFromReference with default options
from Alignment.MuonAlignmentAlgorithms.MuonAlignmentFromReference_cfi import *
looper.algoConfig = MuonAlignmentFromReference

### Diagnostic histograms
MuonAlignmentFromReferenceTFileService = cms.Service("TFileService", fileName = cms.string("MuonAlignmentFromReference.root"))

### Input geometry database
looper.applyDbAlignment = cms.untracked.bool(True)
from CondCore.DBCommon.CondDBSetup_cfi import *
MuonAlignmentFromReferenceInputDB = cms.ESSource("PoolDBESSource",
                                                  CondDBSetup,
                                                  connect = cms.string("sqlite_file:MuonAlignmentFromReference_inputdb.db"),
                                                  toGet = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
                                                                    cms.PSet(record = cms.string("DTAlignmentErrorRcd"), tag = cms.string("DTAlignmentErrorRcd")),
                                                                    cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
                                                                    cms.PSet(record = cms.string("CSCAlignmentErrorRcd"), tag = cms.string("CSCAlignmentErrorRcd"))))
es_prefer_MuonAlignmentFromReferenceInputDB = cms.ESPrefer("PoolDBESSource", "MuonAlignmentFromReferenceInputDB")

### Output geometry database
looper.saveToDB = cms.bool(True)
from CondCore.DBCommon.CondDBSetup_cfi import *
PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  CondDBSetup,
                                  connect = cms.string("sqlite_file:MuonAlignmentFromReference_outputdb.db"),
                                  toPut = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
                                                    cms.PSet(record = cms.string("DTAlignmentErrorRcd"), tag = cms.string("DTAlignmentErrorRcd")),
                                                    cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
                                                    cms.PSet(record = cms.string("CSCAlignmentErrorRcd"), tag = cms.string("CSCAlignmentErrorRcd"))))
