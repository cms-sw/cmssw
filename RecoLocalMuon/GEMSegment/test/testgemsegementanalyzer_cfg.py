import FWCore.ParameterSet.Config as cms

process = cms.Process("TestGEMSegment")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
# process.load('Configuration.Geometry.GeometryExtended2023HGCalMuonReco_cff')

# CSCGeometry depends on alignment ==> necessary to provide GlobalPositionRecord
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi") 
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
# process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:out_local_reco_gemsegment.root'
        # 'file:out_local_reco_noise_gemsegment.root'
        # 'file:out_local_reco_gemsegment_5000evt.root'
        # 'file:out_local_reco_gemsegment_allevts.root'
        # 'file:out_local_reco_test_gemsegment.root'
    )
)

process.gemseg = cms.EDAnalyzer('TestGEMSegmentAnalyzer',
                              # ----------------------------------------------------------------------
                              RootFileName = cms.untracked.string("TestGEMSegmentHistograms.root"),
                              # RootFileName = cms.untracked.string("TestGEMSegmentHistograms_Noise.root"),
                              # RootFileName = cms.untracked.string("TestGEMSegmentHistograms_5000evt.root"),
                              # RootFileName = cms.untracked.string("TestGEMSegmentHistograms_Test.root"),
                              # ----------------------------------------------------------------------
                              printSegmntInfo = cms.untracked.bool(False),
                              printResidlInfo = cms.untracked.bool(False),
                              printSimHitInfo = cms.untracked.bool(False),
                              printEventOrder = cms.untracked.bool(False),
                              # ----------------------------------------------------------------------

)

process.p = cms.Path(process.gemseg)
