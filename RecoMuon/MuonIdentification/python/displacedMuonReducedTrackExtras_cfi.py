import FWCore.ParameterSet.Config as cms

displacedMuonReducedTrackExtras = cms.EDProducer('MuonReducedTrackExtraProducer',
  muonTag = cms.InputTag('displacedMuons'),
  trackExtraTags = cms.VInputTag(
    'displacedTracks',
    'displacedGlobalMuons'
  ),
  trackExtraAssocs = cms.VInputTag(),
  pixelClusterTag = cms.InputTag('siPixelClusters'),
  stripClusterTag = cms.InputTag('siStripClusters'),
  cut = cms.string('pt > 3.0'),
  outputClusters = cms.bool(True),
  mightGet = cms.optional.untracked.vstring
)
