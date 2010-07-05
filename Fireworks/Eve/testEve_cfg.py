import FWCore.ParameterSet.Config as cms

process = cms.Process("DISPLAY")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

### Expects test.root in current directory.
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring('file:test.root')
)

process.TGeoFromDddService = cms.Service(
    "TGeoFromDddService",
    verbose = cms.untracked.bool(False),
    level   = cms.untracked.int32(11)
)

process.EveService = cms.Service("EveService")

# process.maxEvents = cms.untracked.PSet(
#         input = cms.untracked.int32(1)
#         )

process.dump = cms.EDAnalyzer(
    "DummyEvelyser",
    tracks = cms.untracked.InputTag("generalTracks")
)

process.p = cms.Path(process.dump)
