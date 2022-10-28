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
	CSCTimingParameters = dict(PruneCut = 9999)),
    selectHighPurity = False,
    minPt = 0.5
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
        CSCTimingParameters = dict(PruneCut = 9999)),
    selectHighPurity = False,
    minPt = 0.5
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
from RecoMuon.MuonIdentification.displacedMuons_cfi import *
from RecoMuon.MuonIsolation.muonPFIsolation_cff import *
from RecoMuon.MuonIdentification.muonReducedTrackExtras_cfi import *
from RecoMuon.MuonIdentification.displacedMuonReducedTrackExtras_cfi import *

# clusters are not present in fastsim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(muonReducedTrackExtras, outputClusters = False)

# cluster collections are different in phase 2
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(muonReducedTrackExtras, outputClusters = False)
phase2_tracker.toModify(displacedMuonReducedTrackExtras, outputClusters = False)

muonshighlevelrecoTask = cms.Task(muonPFIsolationTask,displacedMuonPFIsolationTask,muons,displacedMuons,muonReducedTrackExtras, displacedMuonReducedTrackExtras)
muonshighlevelreco = cms.Sequence(muonshighlevelrecoTask)

# displaced sequences do not run in fastsim
fastSim.toReplaceWith(muonshighlevelrecoTask,muonshighlevelrecoTask.copyAndExclude([displacedMuonPFIsolationTask,displacedMuons,displacedMuonReducedTrackExtras]))
