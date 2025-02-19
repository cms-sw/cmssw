import FWCore.ParameterSet.Config as cms

# module for fine delay analysis, stored in fake digis to comply with the digi input format of the Commissioning Source
siStripFineDelayHit = cms.EDProducer("SiStripFineDelayHit",
    # general parameters
    cosmic = cms.bool(True),
    MagneticField = cms.bool(False),
    #string mode = "DelayScan"
    ClustersLabel = cms.InputTag("siStripClusters"),
    NoTracking = cms.bool(False),
    # with tracks
    TrajInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    MaxTrackAngle = cms.double(45.0),
    MinTrackMomentum = cms.double(0.0),
    MaxClusterDistance = cms.double(2.0),
    TracksLabel = cms.InputTag("cosmictrackfinder"),
    SeedsLabel = cms.InputTag("cosmicseedfinder"),
    # to avoid the cluster threshold
    NoClustering = cms.bool(True),
    ExplorationWindow = cms.uint32(10),
    DigiLabel = cms.InputTag("siStripZeroSuppression","VirginRaw"),
    # the label for EventSummary
    InputModuleLabel = cms.InputTag("FedChannelDigis")
)
