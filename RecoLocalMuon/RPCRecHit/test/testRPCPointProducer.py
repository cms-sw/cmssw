import FWCore.ParameterSet.Config as cms
process = cms.Process("OwnParticles")

process.load('Configuration.Geometry.GeometryExtended2018Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2018_cff')
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.GlobalTag.globaltag = "124X_dataRun3_Express_v5"


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2000) )

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(

# local copy for test only:
'file:/eos/cms/store/group/dpg_rpc/comm_rpc/Sandbox/mileva/testRPCMon2022/1cb9263a-550d-4e86-92f3-f01574867137.root'
)
)

process.load("RecoLocalMuon.RPCRecHit.rpcPointProducer_cff")
process.rpcPointProducer = cms.EDProducer('RPCPointProducer',
  incldt = cms.bool(True),
  inclcsc = cms.bool(True),
  incltrack =  cms.bool(False),
  debug = cms.bool(False),
  rangestrips = cms.double(4.),
  rangestripsRB4 = cms.double(4.),
  MinCosAng = cms.double(0.85),
  MaxD = cms.double(80.0),
  MaxDrb4 = cms.double(150.0),
  ExtrapolatedRegion = cms.double(0.6), # in stripl/2 in Y and stripw*nstrips/2 in X
  cscSegments = cms.InputTag('dTandCSCSegmentsinTracks','SelectedCscSegments','OwnParticles'),
  dt4DSegments = cms.InputTag('dTandCSCSegmentsinTracks','SelectedDtSegments','OwnParticles'),
  tracks = cms.InputTag("standAloneMuons"),
  TrackTransformer = cms.PSet(
      DoPredictionsOnly = cms.bool(False),
      Fitter = cms.string('KFFitterForRefitInsideOut'),
      TrackerRecHitBuilder = cms.string('WithTrackAngle'),
      Smoother = cms.string('KFSmootherForRefitInsideOut'),
      MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
      RefitDirection = cms.string('alongMomentum'),
      RefitRPCHits = cms.bool(False),
      Propagator = cms.string('SmartPropagatorAnyRKOpposite')
  ),
  minBX = cms.int32(-2),
  maxBX = cms.int32(2)
)

process.p = cms.Path(process.rpcPointProducer)

