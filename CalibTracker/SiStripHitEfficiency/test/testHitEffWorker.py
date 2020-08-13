import FWCore.ParameterSet.Config as cms

process = cms.Process("HitEff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')  

process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/0F7B8753-64A4-8D45-ACD9-146FDB974B3E.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280001/549A26F5-FD43-8D45-A98E-95A0CDF0D233.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/63FB32DC-FAAB-B046-B5C4-90C95CEBABDD.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/5546225F-695A-0343-BBCB-084DFD06B2A5.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/CDD33600-25B8-DC48-9E56-8DC7A48030B9.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/8D4AC402-4D70-454C-9D23-0F819BE93375.root"
    ))

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.refitTracks = process.TrackRefitter.clone(src=cms.InputTag("ALCARECOSiStripCalMinBias"))
tracks = cms.InputTag("refitTracks")

process.hiteff = cms.EDProducer("SiStripHitEfficiencyWorker",
    lumiScalers=cms.InputTag("scalersRawToDigi"),
    addLumi = cms.untracked.bool(False),
    commonMode=cms.InputTag("siStripDigis", "CommonMode"),
    addCommonMode=cms.untracked.bool(False),
    combinatorialTracks = tracks,
    trajectories        = tracks,
    siStripClusters     = cms.InputTag("siStripClusters"),
    siStripDigis        = cms.InputTag("siStripDigis"),
    trackerEvent        = cms.InputTag("MeasurementTrackerEvent"),
    # part 2
    Layer = cms.int32(0), # =0 means do all layers
    Debug = cms.bool(False),
    # do not cut on the total number of tracks
    cutOnTracks = cms.untracked.bool(True),
    trackMultiplicity = cms.untracked.uint32(100),
    # use or not first and last measurement of a trajectory (biases), default is false
    useFirstMeas = cms.untracked.bool(False),
    useLastMeas = cms.untracked.bool(False),
    useAllHitsFromTracksWithMissingHits = cms.untracked.bool(False)
    )

process.load("DQM.SiStripCommon.TkHistoMap_cff")

process.allPath = cms.Path(process.MeasurementTrackerEvent*process.offlineBeamSpot*process.refitTracks*process.hiteff)

# save the DQM plots in the DQMIO format
process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
            fileName = cms.untracked.string("DQM.root")
            )
process.HitEffOutput = cms.EndPath(process.dqmOutput)

process.MessageLogger = cms.Service(
    "MessageLogger",
    destinations = cms.untracked.vstring(
        "log_tkhistomap"
        ),
    log_tkhistomap = cms.untracked.PSet(
        threshold = cms.untracked.string("DEBUG"),
        default = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
        )
    ),
    debugModules = cms.untracked.vstring("hiteff"),
    categories=cms.untracked.vstring("TkHistoMap")
)
