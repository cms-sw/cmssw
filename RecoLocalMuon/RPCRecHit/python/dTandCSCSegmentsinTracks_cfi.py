import FWCore.ParameterSet.Config as cms

dTandCSCSegmentsinTracks = cms.EDProducer("DTandCSCSegmentsinTracks",
  
#  cscSegments = cms.InputTag('hltCscSegments'),
#  dt4DSegments = cms.InputTag('hltDt4DSegments'),
  cscSegments = cms.InputTag('cscSegments'),
  dt4DSegments = cms.InputTag('dt4DSegments'),

  tracks = cms.InputTag("standAloneMuons","UpdatedAtVtx"),


)
