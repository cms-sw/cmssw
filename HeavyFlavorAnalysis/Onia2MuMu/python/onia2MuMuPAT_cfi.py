import FWCore.ParameterSet.Config as cms

onia2MuMuPAT = cms.EDProducer('Onia2MuMuPAT',
  muons = cms.InputTag("patMuons"),
  beamSpotTag = cms.InputTag("offlineBeamSpot"),
  primaryVertexTag = cms.InputTag("offlinePrimaryVertices"),
  higherPuritySelection = cms.string("isGlobalMuon"), ## At least one muon must pass this selection
  lowerPuritySelection  = cms.string("isGlobalMuon"), ## BOTH muons must pass this selection
  dimuonSelection  = cms.string("2 < mass && abs(daughter('muon1').innerTrack.dz - daughter('muon2').innerTrack.dz) < 25"), ## The dimuon must pass this selection before vertexing
  addCommonVertex = cms.bool(True), ## Embed the full reco::Vertex out of the common vertex fit
  addMuonlessPrimaryVertex = cms.bool(False), ## Embed the primary vertex re-made from all the tracks except the two muons
  addMCTruth = cms.bool(True),      ## Add the common MC mother of the two muons, if any
  resolvePileUpAmbiguity = cms.bool(True)   ## Order PVs by their vicinity to the J/psi vertex, not by sumPt                            
)
