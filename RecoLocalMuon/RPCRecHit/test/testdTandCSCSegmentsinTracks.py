import FWCore.ParameterSet.Config as cms

##process = cms.Process("RPCPointProducer")
process = cms.Process("AsociatedSegmentsProducer")


process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")



process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'START3X_V18::All'
process.GlobalTag.globaltag = 'GR_R_42_V22::All'



process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#	'rfio:/castor/cern.ch/user/g/ggeorge/mileva/reco_oldCls/DYToMuMu_M60_RECO_Eta16_oldCLS_21.root'
	'file:/tmp/carrillo/DYToMuMu_M60_RECO_Eta16_oldCLS_21.root'
        )                           
)

process.load("RecoLocalMuon.RPCRecHit.dTandCSCSegmentsinTracks_cfi")

process.out = cms.OutputModule("PoolOutputModule",
 fileName = cms.untracked.string('/tmp/carrillo/outs/WithNewContainer.root')
)
  
process.p = cms.Path(process.dTandCSCSegmentsinTracks)

process.e = cms.EndPath(process.out)
