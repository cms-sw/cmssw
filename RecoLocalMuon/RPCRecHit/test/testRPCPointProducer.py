import FWCore.ParameterSet.Config as cms

##process = cms.Process("RPCPointProducer")
process = cms.Process("OWNPARTICLES")

process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_3XY_V15::All"

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#	'file:/tmp/carrillo/outs/WithNewContainer.root'
	'rfio:/castor/cern.ch/user/j/jgomezca/NewRun1Container/_RPCMonitor_Run2011B-v1_RAW/180250/NewRun1Container-180250.E843AA1E-FD02-E111-81D3-BCAEC5329714.root'
        )                           
)


#process.load("RecoLocalMuon.RPCRecHit.rpcPointProducer_cfi")
process.load("RecoLocalMuon.RPCRecHit.rpcPointProducerTrackSegments_cfi")

#process.rpcPointProducer.debug = cms.untracked.bool(True),

process.out = cms.OutputModule("PoolOutputModule",
  outputCommands = cms.untracked.vstring('drop *',
        'keep *_dt4DSegments_*_*',
        'keep *_cscSegments_*_*',
        'keep *_SelectedCscSegments_*_*',
        'keep *_SelectedDtSegments_*_*',
        'keep *_rpcPointProducer_*_*',
        'keep *_rpcRecHits_*_*',
        'keep *_standAloneMuons_*_*',
        'keep *_cosmicMuons_*_*',
        'keep *_globalMuons_*_*'),
 fileName = cms.untracked.string('/tmp/carrillo/outs/PointsFromSelectedSegments.root')
)
  
process.p = cms.Path(process.rpcPointProducer)

process.e = cms.EndPath(process.out)
