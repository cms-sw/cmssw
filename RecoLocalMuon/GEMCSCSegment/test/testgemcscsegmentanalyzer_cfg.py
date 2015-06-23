import FWCore.ParameterSet.Config as cms

process = cms.Process("TestGEMCSCSegment")
process.load("FWCore.MessageService.MessageLogger_cfi")

# process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
# process.load('Configuration.Geometry.GeometryExtended2019_cff')
process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023_cff')
process.load('Configuration.EventContent.EventContent_cff')
# process.load("Geometry.GEMGeometry.gemGeometry_cfi")
# process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
# process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixDTAlignmentConditions
process = fixDTAlignmentConditions(process)
from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixCSCAlignmentConditions
process = fixCSCAlignmentConditions(process)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        # 'file:/lustre/cms/store/user/radogna/GEMCSCSegment/FullDigi_withNewSSegm/out_rec_gemcsc.test10.root'
        # 'file:out_reco10_paranormal.root'
        # 'file:out_rec_gemcsc.test10_xcommit.root'
        # 'file:out_reco10_overlap.root'
        # 'file:out_reco_overlap.root'
        # 'file:out_rereco_overlap.root'
        # 'file:SingleMuPt10_RECO_upgrade2019_lowstat.root'
        # 'file:out_rereco_gemcscsegment.root'
        'file:out_rereco_gemcscsegment_75_2023_100GeV_1000evts.root'
    )
)

process.gemcscs = cms.EDAnalyzer('TestGEMCSCSegmentAnalyzer',
                              RootFileName = cms.untracked.string("TestGEMCSCSegmentHistograms_75_2023_100GeV_1000evts.root"),
                              Debug = cms.untracked.bool(True),
)

process.p = cms.Path(process.gemcscs)
