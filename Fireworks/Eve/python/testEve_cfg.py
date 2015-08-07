import FWCore.ParameterSet.Config as cms

process = cms.Process("DISPLAY")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

### Expects test.root in current directory.
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring('file:/home/alja/cms-dev/7.3/RelValZEE-reco.root')
)

# process.maxEvents = cms.untracked.PSet(
#         input = cms.untracked.int32(1)
#         )

### For running on pre 3.6 files the current needed to determine the
### magnetic field is taken from Conditions DB.
# process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
### specify tag:
# process.GlobalTag.globaltag = 'START36_V10::All'
### or use auto-cond:
# from Configuration.AlCa.autoCond import autoCond
# process.GlobalTag.globaltag = autoCond['mc']

### Request EveService
process.EveService = cms.Service("EveService")

### Extractor of geometry needed to display it in Eve.
### Required for "DummyEvelyser".
process.add_( cms.ESProducer(
        "TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(8)
))

process.dump = cms.EDAnalyzer(
    "DummyEvelyser",
    tracks = cms.untracked.InputTag("generalTracks")
)

process.p = cms.Path(process.dump)
