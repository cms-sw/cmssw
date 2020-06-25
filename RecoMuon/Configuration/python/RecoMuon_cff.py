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
muonsFromCosmics = RecoMuon.MuonIdentification.muons1stStep_cfi.muons1stStep.clone(
    inputCollectionLabels = ['cosmicMuons'],
    inputCollectionTypes = ['outer tracks'],
    fillIsolation = False,
    fillGlobalTrackQuality = False,
    fillGlobalTrackRefits = False,
    TrackAssociatorParameters = dict(DTRecSegment4DCollectionLabel = 'dt4DCosmicSegments'),
    TrackExtractorPSet = dict(inputTrackCollection = 'cosmicMuons'),
    TimingFillerParameters = dict(
	MatchParameters = dict(DTsegments = 'dt4DCosmicSegments'),
	DTTimingParameters = dict(PruneCut = 9999),
	CSCTimingParameters = dict(PruneCut = 9999))
)

#add regional cosmic tracks here
muoncosmicreco2legsSTATask = cms.Task(CosmicMuonSeed,cosmicMuons)
muoncosmicreco2legsSTA = cms.Sequence(muoncosmicreco2legsSTATask)
muoncosmicreco2legsHighLevelTask = cms.Task(muonsFromCosmics)
muoncosmicreco2legsHighLevel = cms.Sequence(muoncosmicreco2legsHighLevelTask)

# 1 Leg type
# Stand alone muon track producer
cosmicMuons1Leg = cosmicMuons.clone(
    MuonSeedCollectionLabel = 'CosmicMuonSeed',
    TrajectoryBuilderParameters = dict(
	BuildTraversingMuon = True, 
	Strict1Leg = True, 
	DTRecSegmentLabel = 'dt4DCosmicSegments')
)

# Muon Id producer
muonsFromCosmics1Leg = muons1stStep.clone(
    inputCollectionLabels = ['cosmicMuons1Leg'],
    inputCollectionTypes = ['outer tracks'],
    fillIsolation = False,
    fillGlobalTrackQuality = False,
    fillGlobalTrackRefits = False,
    TrackAssociatorParameters = dict(DTRecSegment4DCollectionLabel = 'dt4DCosmicSegments'),
    TrackExtractorPSet = dict(inputTrackCollection = 'cosmicMuons1Leg'),
    TimingFillerParameters = dict(
        MatchParameters = dict(DTsegments = 'dt4DCosmicSegments'),
        DTTimingParameters = dict(PruneCut = 9999),
        CSCTimingParameters = dict(PruneCut = 9999))
)

muoncosmicreco1legSTATask = cms.Task(CosmicMuonSeed,cosmicMuons1Leg)
muoncosmicreco1legSTA = cms.Sequence(muoncosmicreco1legSTATask)
muoncosmicreco1legHighLevelTask = cms.Task(muonsFromCosmics1Leg)
muoncosmicreco1legHighLevel = cms.Sequence(muoncosmicreco1legHighLevelTask)

muoncosmicrecoTask = cms.Task(muoncosmicreco2legsSTATask,muoncosmicreco1legSTATask)
muoncosmicreco = cms.Sequence(muoncosmicrecoTask)
muoncosmichighlevelrecoTask = cms.Task(muoncosmicreco2legsHighLevelTask,muoncosmicreco1legHighLevelTask,cosmicsMuonIdTask)
muoncosmichighlevelreco = cms.Sequence(muoncosmichighlevelrecoTask)

#### High level sequence (i.e., post PF reconstruction) ###
from RecoMuon.MuonIdentification.muons_cfi import *
from RecoMuon.MuonIsolation.muonPFIsolation_cff import *
muonshighlevelrecoTask = cms.Task(muonPFIsolationTask,muons)
muonshighlevelreco = cms.Sequence(muonshighlevelrecoTask)
