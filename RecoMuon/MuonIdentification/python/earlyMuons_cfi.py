import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIdentification.muons1stStep_cfi import muons1stStep

earlyMuons = muons1stStep.clone(
    inputCollectionTypes = ['inner tracks','outer tracks'],
    inputCollectionLabels = ['earlyGeneralTracks', 'standAloneMuons:UpdatedAtVtx'],
    minP         = 3.0, # was 2.5
    minPt        = 2.0, # was 0.5
    minPCaloMuon = 3.0, # was 1.0
    fillCaloCompatibility = False,
    fillEnergy = False,
    fillGlobalTrackQuality = False,
    fillGlobalTrackRefits  = False,
    fillIsolation = False,
    fillTrackerKink = False,
    TrackAssociatorParameters = dict(
	useHO   = False,
	useEcal = False,
	useHcal = False)
)
earlyDisplacedMuons = earlyMuons.clone(
    inputCollectionLabels = ['earlyGeneralTracks','displacedStandAloneMuons:']
)
