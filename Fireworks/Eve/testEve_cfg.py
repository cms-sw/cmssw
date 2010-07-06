import FWCore.ParameterSet.Config as cms

process = cms.Process("DISPLAY")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

### Expects test.root in current directory.
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring('file:test.root')
)

process.add_( cms.ESProducer(
        "TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(4)
))

process.EveService = cms.Service("EveService")

# process.maxEvents = cms.untracked.PSet(
#         input = cms.untracked.int32(1)
#         )

process.dump = cms.EDAnalyzer(
    "DummyEvelyser",
    tracks = cms.untracked.InputTag("generalTracks")
)

process.p = cms.Path(process.dump)
