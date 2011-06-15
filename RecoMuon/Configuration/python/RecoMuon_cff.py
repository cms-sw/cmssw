import FWCore.ParameterSet.Config as cms

# Standard pp setup
from RecoMuon.Configuration.RecoMuonPPonly_cff import *

########################################################

# Sequence for cosmic reconstruction

# Seed generator
from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *
CosmicMuonSeed.DTRecSegmentLabel = 'dt4DCosmicSegments'


# Stand alone muon track producer
from RecoMuon.CosmicMuonProducer.cosmicMuons_cff import *
cosmicMuons.TrajectoryBuilderParameters.DTRecSegmentLabel = 'dt4DCosmicSegments'

# Global muon track producer
from RecoMuon.CosmicMuonProducer.globalCosmicMuons_cff import *
globalCosmicMuons.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'regionalCosmicTracks'

# Muon Id producer
muonsFromCosmics = RecoMuon.MuonIdentification.muons1stStep_cfi.muons1stStep.clone()

muonsFromCosmics.inputCollectionLabels = ['globalCosmicMuons', 'cosmicMuons', 'regionalCosmicTracks']
muonsFromCosmics.inputCollectionTypes = ['links', 'outer tracks', 'inner tracks' ]
muonsFromCosmics.TrackAssociatorParameters.DTRecSegment4DCollectionLabel = 'dt4DCosmicSegments'
muonsFromCosmics.TrackExtractorPSet.inputTrackCollection = 'regionalCosmicTracks'
muonsFromCosmics.TimingFillerParameters.DTTimingParameters.MatchParameters.DTsegments = 'dt4DCosmicSegments'
muonsFromCosmics.TimingFillerParameters.DTTimingParameters.DTsegments = 'dt4DCosmicSegments' 
muonsFromCosmics.TimingFillerParameters.CSCTimingParameters.MatchParameters.DTsegments = 'dt4DCosmicSegments'
muonsFromCosmics.fillIsolation = False
muonsFromCosmics.fillGlobalTrackQuality = False
muonsFromCosmics.fillGlobalTrackRefits = False

#from RecoTracker.Configuration.RecoTrackerNotStandard_cff import *
#add regional cosmic tracks here
#muoncosmicreco2legs = cms.Sequence(cosmicMuons*regionalCosmicTracksSeq*globalCosmicMuons*muonsFromCosmics)
muoncosmicreco2legsSTA = cms.Sequence(CosmicMuonSeed*cosmicMuons)
muoncosmicreco2legsHighLevel = cms.Sequence(globalCosmicMuons*muonsFromCosmics)

# 1 Leg type

# Stand alone muon track producer
cosmicMuons1Leg = cosmicMuons.clone()
cosmicMuons1Leg.TrajectoryBuilderParameters.BuildTraversingMuon = True
cosmicMuons1Leg.TrajectoryBuilderParameters.Strict1Leg = True
cosmicMuons1Leg.TrajectoryBuilderParameters.DTRecSegmentLabel = 'dt4DCosmicSegments'
cosmicMuons1Leg.MuonSeedCollectionLabel = 'CosmicMuonSeed'

# Global muon track producer
globalCosmicMuons1Leg = globalCosmicMuons.clone()
globalCosmicMuons1Leg.MuonCollectionLabel = 'cosmicMuons1Leg'
globalCosmicMuons1Leg.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'regionalCosmicTracks'

# Muon Id producer
muonsFromCosmics1Leg = muons1stStep.clone()
muonsFromCosmics1Leg.inputCollectionLabels = ['globalCosmicMuons1Leg', 'cosmicMuons1Leg',  'regionalCosmicTracks' ]
muonsFromCosmics1Leg.inputCollectionTypes = ['links', 'outer tracks', 'inner tracks' ]
muonsFromCosmics1Leg.TrackAssociatorParameters.DTRecSegment4DCollectionLabel = 'dt4DCosmicSegments'
muonsFromCosmics1Leg.TrackExtractorPSet.inputTrackCollection = 'regionalCosmicTracks'
muonsFromCosmics1Leg.TimingFillerParameters.DTTimingParameters.MatchParameters.DTsegments = 'dt4DCosmicSegments'
muonsFromCosmics1Leg.TimingFillerParameters.DTTimingParameters.DTsegments = 'dt4DCosmicSegments' 
muonsFromCosmics1Leg.TimingFillerParameters.CSCTimingParameters.MatchParameters.DTsegments = 'dt4DCosmicSegments'
muonsFromCosmics1Leg.fillIsolation = False
muonsFromCosmics1Leg.fillGlobalTrackQuality = False
muonsFromCosmics1Leg.fillGlobalTrackRefits = False

#muoncosmicreco1leg = cms.Sequence(cosmicMuons1Leg*globalCosmicMuons1Leg*muonsFromCosmics1Leg)
muoncosmicreco1legSTA = cms.Sequence(CosmicMuonSeed*cosmicMuons1Leg)
muoncosmicreco1legHighLevel = cms.Sequence(globalCosmicMuons1Leg*muonsFromCosmics1Leg)

#muoncosmicreco = cms.Sequence(CosmicMuonSeed*(muoncosmicreco2legs+muoncosmicreco1leg)*cosmicsMuonIdSequence)
muoncosmicreco = cms.Sequence(muoncosmicreco2legsSTA+muoncosmicreco1legSTA)
muoncosmichighlevelreco = cms.Sequence((muoncosmicreco2legsHighLevel+muoncosmicreco1legHighLevel)*cosmicsMuonIdSequence)


#### High level sequence (i.e., post PF reconstruction) ###

from RecoMuon.MuonIdentification.muons_cfi import *
from RecoMuon.MuonIsolation.muonPFIsolation_cff import *

muonshighlevelreco = cms.Sequence(muonPFIsolationSequence*muons)
