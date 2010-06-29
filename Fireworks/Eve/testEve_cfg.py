import FWCore.ParameterSet.Config as cms

process = cms.Process("DISPLAY")

# process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

# process.load("Geometry.DTGeometry.dtGeometry_cfi")
# process.load("Geometry.CSCGeometry.cscGeometry_cfi")

# process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
# process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

# process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

### Expects test.root in current directory.
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring('file:test.root')
)

process.EveService = cms.Service("EveService")

# process.add_( cms.ESProducer("DisplayGeomFromDDD") )

# process.maxEvents = cms.untracked.PSet(
#         input = cms.untracked.int32(1)
#         )

process.dump = cms.EDAnalyzer(
    "DummyEvelyser",
    tracks = cms.untracked.InputTag("generalTracks")
)

process.p = cms.Path(process.dump)
