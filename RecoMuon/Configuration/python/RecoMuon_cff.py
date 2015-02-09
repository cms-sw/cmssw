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

# Muon Id producer
muonsFromCosmics = RecoMuon.MuonIdentification.muons1stStep_cfi.muons1stStep.clone()
muonsFromCosmics.inputCollectionLabels = ['cosmicMuons']
muonsFromCosmics.inputCollectionTypes = ['outer tracks']
muonsFromCosmics.TrackAssociatorParameters.DTRecSegment4DCollectionLabel = 'dt4DCosmicSegments'
muonsFromCosmics.TrackExtractorPSet.inputTrackCollection = 'cosmicMuons'
muonsFromCosmics.TimingFillerParameters.DTTimingParameters.MatchParameters.DTsegments = 'dt4DCosmicSegments'
muonsFromCosmics.TimingFillerParameters.CSCTimingParameters.MatchParameters.DTsegments = 'dt4DCosmicSegments'
muonsFromCosmics.fillIsolation = False
muonsFromCosmics.fillGlobalTrackQuality = False
muonsFromCosmics.fillGlobalTrackRefits = False

#add regional cosmic tracks here
muoncosmicreco2legsSTA = cms.Sequence(CosmicMuonSeed*cosmicMuons)
muoncosmicreco2legsHighLevel = cms.Sequence(muonsFromCosmics)

# 1 Leg type
# Stand alone muon track producer
cosmicMuons1Leg = cosmicMuons.clone()
cosmicMuons1Leg.TrajectoryBuilderParameters.BuildTraversingMuon = True
cosmicMuons1Leg.TrajectoryBuilderParameters.Strict1Leg = True
cosmicMuons1Leg.TrajectoryBuilderParameters.DTRecSegmentLabel = 'dt4DCosmicSegments'
cosmicMuons1Leg.MuonSeedCollectionLabel = 'CosmicMuonSeed'

# Muon Id producer
muonsFromCosmics1Leg = muons1stStep.clone()
muonsFromCosmics1Leg.inputCollectionLabels = ['cosmicMuons1Leg']
muonsFromCosmics1Leg.inputCollectionTypes = ['outer tracks']
muonsFromCosmics1Leg.TrackAssociatorParameters.DTRecSegment4DCollectionLabel = 'dt4DCosmicSegments'
muonsFromCosmics1Leg.TrackExtractorPSet.inputTrackCollection = 'cosmicMuons1Leg'
muonsFromCosmics1Leg.TimingFillerParameters.DTTimingParameters.MatchParameters.DTsegments = 'dt4DCosmicSegments'
muonsFromCosmics1Leg.TimingFillerParameters.CSCTimingParameters.MatchParameters.DTsegments = 'dt4DCosmicSegments'
muonsFromCosmics1Leg.fillIsolation = False
muonsFromCosmics1Leg.fillGlobalTrackQuality = False
muonsFromCosmics1Leg.fillGlobalTrackRefits = False
muoncosmicreco1legSTA = cms.Sequence(CosmicMuonSeed*cosmicMuons1Leg)
muoncosmicreco1legHighLevel = cms.Sequence(muonsFromCosmics1Leg)

muoncosmicreco = cms.Sequence(muoncosmicreco2legsSTA+muoncosmicreco1legSTA)
muoncosmichighlevelreco = cms.Sequence((muoncosmicreco2legsHighLevel+muoncosmicreco1legHighLevel)*cosmicsMuonIdSequence)
#### High level sequence (i.e., post PF reconstruction) ###
from RecoMuon.MuonIdentification.muons_cfi import *
from RecoMuon.MuonIsolation.muonPFIsolation_cff import *
muonshighlevelreco = cms.Sequence(muonPFIsolationSequence*muons) 
