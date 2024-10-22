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
process.rpcPointProducer.ExtrapolatedRegion = 0.6 # in stripl/2 in Y and stripw*nstrips/2 in X
process.rpcPointProducer.cscSegments = ('dTandCSCSegmentsinTracks','SelectedCscSegments','OwnParticles')
process.rpcPointProducer.dt4DSegments = ('dTandCSCSegmentsinTracks','SelectedDtSegments','OwnParticles')

process.p = cms.Path(process.rpcPointProducer)

