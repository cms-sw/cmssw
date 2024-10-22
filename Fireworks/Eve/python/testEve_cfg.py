import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("DISPLAY")


options = VarParsing.VarParsing ()
options.register ('file',
                  "xxx", # default value
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "xrootd URL")


options.parseArguments()


process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")


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

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring('file:' + options.file)
)

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
