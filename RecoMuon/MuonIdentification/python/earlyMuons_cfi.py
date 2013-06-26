import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIdentification.muons1stStep_cfi import muons1stStep

earlyMuons = muons1stStep.clone(
    inputCollectionTypes = cms.vstring('inner tracks','outer tracks'),
    inputCollectionLabels = cms.VInputTag(cms.InputTag("earlyGeneralTracks"),cms.InputTag("standAloneMuons","UpdatedAtVtx")),
    minP         = 3.0, # was 2.5
    minPt        = 2.0, # was 0.5
    minPCaloMuon = 3.0, # was 1.0
    fillCaloCompatibility = False,
    fillEnergy = False,
    fillGlobalTrackQuality = False,
    fillGlobalTrackRefits  = False,
    fillIsolation = False,
    fillTrackerKink = False)

earlyMuons.TrackAssociatorParameters.useHO   = cms.bool(False)
earlyMuons.TrackAssociatorParameters.useEcal = cms.bool(False)
earlyMuons.TrackAssociatorParameters.useHcal = cms.bool(False)


