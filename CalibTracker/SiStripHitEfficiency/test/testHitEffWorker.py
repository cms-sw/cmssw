import FWCore.ParameterSet.Config as cms

process = cms.Process("HitEff")
process.load("Configuration/StandardSequences/MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')  

process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/F3F28561-1B7E-0444-810D-A119929B4896.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/80508514-FC4F-5E48-84BD-A84EF28EEAE3.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/86DB491C-55E8-9C45-A748-632065719E7B.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/7D9F8691-6521-5247-88A8-97A9FD11EBB6.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280002/2185F94D-CE60-4541-A174-281FEC1217F6.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/5196D8AE-F413-4344-B14D-40CEB7B57736.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/478C4A87-9779-4043-8EF1-5F0938FC9715.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280002/77E53E14-7349-7C4D-926E-0F7151278A53.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280002/5425331D-22AC-AC4D-A057-F031396B712B.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/F05C07EA-AFA7-184C-8E88-7D1D0B6754FA.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/B559B4C1-A0EE-3444-B54E-0B6BD74E9C81.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/8DCD5AED-9053-0649-AF55-6FC644C84A26.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/232BDB85-A055-0641-BF22-28FF482621F5.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/76328DC5-3778-5E4B-BEC1-B1FCD8305D3C.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280001/A26F1C97-DF47-0248-8176-5EA46DC7083F.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/9831870B-57EF-4745-A70A-488A5E2D16FD.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/BC20D7BE-13C7-CD46-9748-09D79AEED760.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/EE51D346-A691-8744-8541-9B028E4BE5C0.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280002/001B8CC6-BBE5-174A-9D83-4126F609020F.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/9393F683-B456-414C-9494-A54BB8E60963.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/F3D0305D-CBB9-4946-8B86-CD0E97B947DC.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/4091B173-B4B9-6648-8474-B10500EB15F4.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/03F44D76-2079-C34C-8E5B-5214B22FE2DC.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280002/4688663E-2C2A-5D43-A0A6-7B1F3FDE7352.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/98DAF8BF-CA7D-1C4D-A6CD-7ACF1441F03D.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/9F86DA7C-1650-214B-A07E-1081DC1A3229.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280002/1D650158-4FD2-7345-9BC5-5995B61D9E95.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/3FEA0EC7-0301-364A-A356-D5223D970579.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/2CB86DCB-22E4-8245-9477-886B8C553D5E.root",
    "file:/eos/cms/store/data/Run2018D/ZeroBias/ALCARECO/SiStripCalMinBias-12Nov2019_UL2018-v3/280000/F5BBBBCF-2EC7-F54D-8A9C-A557B12F77C1.root"
    ))

#process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(15))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))

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
    Debug = cms.bool(True),
    # do not cut on the total number of tracks
    cutOnTracks = cms.untracked.bool(True),
    trackMultiplicity = cms.untracked.uint32(100),
    # use or not first and last measurement of a trajectory (biases), default is false
    useFirstMeas = cms.untracked.bool(False),
    useLastMeas = cms.untracked.bool(False),
    useAllHitsFromTracksWithMissingHits = cms.untracked.bool(False)
    )

process.load("DQM.SiStripCommon.TkHistoMap_cff")

## OLD HITEFF
from CalibTracker.SiStripHitEfficiency.SiStripHitEff_cff import anEff
process.anEff = anEff
process.anEff.Debug = True
process.anEff.combinatorialTracks = tracks
process.anEff.trajectories = tracks
process.TFileService = cms.Service("TFileService",
        fileName = cms.string('HitEffTree.root')
)
## END OLD HITEFF

## TODO double-check in main CalibTree config if hiteff also takes refitted tracks
process.allPath = cms.Path(process.MeasurementTrackerEvent*process.offlineBeamSpot*process.refitTracks
        *process.anEff
        *process.hiteff)

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
    debugModules = cms.untracked.vstring("hiteff", "anEff"),
    categories=cms.untracked.vstring("TkHistoMap", "SiStripHitEfficiency:HitEff", "SiStripHitEfficiency")
)
