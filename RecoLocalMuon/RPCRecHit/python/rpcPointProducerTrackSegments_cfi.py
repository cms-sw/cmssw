import FWCore.ParameterSet.Config as cms

rpcPointProducer = cms.EDProducer("RPCPointProducer",
  incldt = cms.untracked.bool(True),
  inclcsc = cms.untracked.bool(True),

  debug = cms.untracked.bool(True),

  rangestrips = cms.untracked.double(4.),
  rangestripsRB4 = cms.untracked.double(4.),
  MinCosAng = cms.untracked.double(0.85),
  MaxD = cms.untracked.double(80.0),
  MaxDrb4 = cms.untracked.double(150.0),
  ExtrapolatedRegion = cms.untracked.double(0.5), #in stripl/2 in Y and stripw*nstrips/2 in X

#  cscSegments = cms.InputTag('hltCscSegments'),
#  dt4DSegments = cms.InputTag('hltDt4DSegments'),
#  cscSegments = cms.InputTag('cscSegments'),
#  dt4DSegments = cms.InputTag('dt4DSegments'),
   dt4DSegments = cms.InputTag("dTandCSCSegmentsinTracks","SelectedDtSegments"),
   cscSegments = cms.InputTag("dTandCSCSegmentsinTracks","SelectedCscSegments"),

)
