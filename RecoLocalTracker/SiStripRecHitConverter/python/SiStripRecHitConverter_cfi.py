import FWCore.ParameterSet.Config as cms

siStripMatchedRecHits = cms.EDProducer("SiStripRecHitConverter",
    Regional = cms.bool(False),
    ClusterProducer    = cms.InputTag('siStripClusters'),
    LazyGetterProducer = cms.InputTag('SiStripRawToClustersFacility'), # used if Regional is True
    StripCPE            = cms.ESInputTag('StripCPEfromTrackAngleESProducer:StripCPEfromTrackAngle'),
    Matcher             = cms.ESInputTag('SiStripRecHitMatcherESProducer:StandardMatcher'),
    siStripQualityLabel = cms.ESInputTag(''),
    useSiStripQuality = cms.bool(False),
    MaskBadAPVFibers  = cms.bool(False),
    rphiRecHits    = cms.string('rphiRecHit'),
    stereoRecHits  = cms.string('stereoRecHit'),
    matchedRecHits = cms.string('matchedRecHit'),
    VerbosityLevel = cms.untracked.int32(1)
)
