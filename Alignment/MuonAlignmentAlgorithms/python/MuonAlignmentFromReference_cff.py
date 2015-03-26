import FWCore.ParameterSet.Config as cms

### General track re-fitting includes
### (don't load dtGeometry_cfi or cscGeometry_cfi because it's provided by AlignmentProducer)
from Configuration.StandardSequences.Services_cff import *
from Configuration.StandardSequences.GeometryExtended_cff import *
from Configuration.StandardSequences.MagneticField_cff import *
from RecoTracker.Configuration.RecoTracker_cff import *
del DTGeometryESModule
del CSCGeometryESModule

### Track refitter for global collisions muons
from TrackingTools.TrackRefitter.globalMuonTrajectories_cff import *
MuonAlignmentFromReferenceGlobalMuonRefit = globalMuons.clone()
MuonAlignmentFromReferenceGlobalMuonRefit.Tracks = cms.InputTag("ALCARECOMuAlCalIsolatedMu:GlobalMuon")
MuonAlignmentFromReferenceGlobalMuonRefit.TrackTransformer.RefitRPCHits = cms.bool(False)

### Track refitter for global cosmic muons
from TrackingTools.TrackRefitter.globalCosmicMuonTrajectories_cff import *
MuonAlignmentFromReferenceGlobalCosmicRefit = globalCosmicMuons.clone()
MuonAlignmentFromReferenceGlobalCosmicRefit.Tracks = cms.InputTag("ALCARECOMuAlGlobalCosmics:GlobalMuon")
MuonAlignmentFromReferenceGlobalCosmicRefit.TrackTransformer.RefitRPCHits = cms.bool(False)

### for Tracker muon re-reco
from RecoMuon.Configuration.RecoMuon_cff import *
newmuons = muons.clone(
  inputCollectionTypes = cms.vstring("inner tracks"),
  #inputCollectionLabels = cms.VInputTag(cms.InputTag("generalTracks")),
  inputCollectionLabels = cms.VInputTag(cms.InputTag("refittedGeneralTracks")),
  fillIsolation = cms.bool(False),
)

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
                                                                    cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"), tag = cms.string("DTAlignmentErrorExtendedRcd")),
                                                                    cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
                                                                    cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"), tag = cms.string("CSCAlignmentErrorExtendedRcd"))))
es_prefer_MuonAlignmentFromReferenceInputDB = cms.ESPrefer("PoolDBESSource", "MuonAlignmentFromReferenceInputDB")

### Output geometry database
looper.saveToDB = cms.bool(True)
from CondCore.DBCommon.CondDBSetup_cfi import *
PoolDBOutputService = cms.Service("PoolDBOutputService",
                                  CondDBSetup,
                                  connect = cms.string("sqlite_file:MuonAlignmentFromReference_outputdb.db"),
                                  toPut = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
                                                    cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"), tag = cms.string("DTAlignmentErrorExtendedRcd")),
                                                    cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
                                                    cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"), tag = cms.string("CSCAlignmentErrorExtendedRcd"))))
