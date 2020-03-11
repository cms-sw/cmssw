import FWCore.ParameterSet.Config as cms

process = cms.Process("MuonAlignmentMonitor")

process.load("Configuration.Geometry.GeometryPilot2_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# ideal geometry and interface
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi")

process.source = cms.Source("PoolSource",
    #AlCaReco File
    fileNames = cms.untracked.vstring('MuAlZMuMu_ALCARECO.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("Alignment.OfflineValidation.MuonAlignmentAnalyzer_cfi")
# InputTags for AlCaRecoMuon format (e.g.)
process.MuonAlignmentMonitor.StandAloneTrackCollectionTag = "ALCARECOMuAlZMuMu:StandAlone"
process.MuonAlignmentMonitor.GlobalMuonTrackCollectionTag = "ALCARECOMuAlZMuMu:GlobalMuon"

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MuonAlignmentMonitor.root')
)

process.p = cms.Path(process.MuonAlignmentMonitor)
process.GlobalTag.globaltag = 'IDEAL_V9::All'


