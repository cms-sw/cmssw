import FWCore.ParameterSet.Config as cms

######################## Cosmic Reco #############################

## Full detector ##

# Seed generator
from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *

# Stand alone muon track producer
from RecoMuon.CosmicMuonProducer.cosmicMuons_cff import *

# Global muon track producer
from RecoMuon.CosmicMuonProducer.globalCosmicMuons_cff import *
globalCosmicMuons.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5LHCNavigation'

# Muon Id producer
from RecoMuon.MuonIdentification.muonIdProducerSequence_cff import *

muons = muons1stStep.clone(
    inputCollectionLabels = ['ctfWithMaterialTracksP5LHCNavigation', 
                             'globalCosmicMuons', 
                             'cosmicMuons', 
                             'tevMuons:firstHit',
                             'tevMuons:picky',
                             'tevMuons:dyt'],

    inputCollectionTypes = ['inner tracks', 
                            'links', 
                            'outer tracks', 
                            'tev firstHit', 
                            'tev picky', 
                            'tev dyt'],
    fillIsolation = True,
    fillGlobalTrackQuality = True,
    # need to modify track selection as well (not clear to what)
    TrackExtractorPSet = dict(inputTrackCollection = 'ctfWithMaterialTracksP5LHCNavigation'),
    # set wide cone until the code is made to compute this wrt CalIntersection
    CaloExtractorPSet = dict(CenterConeOnCalIntersection = True, DR_Max = 1.0),
    TimingFillerParameters = dict(
	DTTimingParameters = dict(PruneCut = 9999),
	CSCTimingParameters = dict(PruneCut = 9999))
)

#similar to what's in pp configuration
muonsFromCosmics = muons1stStep.clone(
    inputCollectionLabels = ['cosmicMuons'],
    inputCollectionTypes = ['outer tracks'],
    fillIsolation = False,
    fillGlobalTrackQuality = False,
    fillGlobalTrackRefits = False,
    TrackExtractorPSet = dict(inputTrackCollection = 'cosmicMuons'),
    TimingFillerParameters = dict(
	DTTimingParameters = dict(PruneCut = 9999),
	CSCTimingParameters = dict(PruneCut = 9999))
)

## Sequences
# Stand Alone Tracking
STAmuontrackingforcosmicsTask = cms.Task(CosmicMuonSeed,cosmicMuons)
STAmuontrackingforcosmics = cms.Sequence(STAmuontrackingforcosmicsTask)

# Stand Alone Tracking plus global tracking
muontrackingforcosmicsTask = cms.Task(STAmuontrackingforcosmicsTask,globalCosmicMuons)
muontrackingforcosmics = cms.Sequence(muontrackingforcosmicsTask)

# Muon Isolation sequence
from RecoMuon.MuonIsolationProducers.muIsolation_cff import *
# muisodeposits based on "muons"
# we are using copy extractors now
muIsoDepositTk.inputTags = ['muons:tracker']
muIsoDepositJets.inputTags = ['muons:jets']
muIsoDepositCalByAssociatorTowers.inputTags = ['muons:ecal', 'muons:hcal', 'muons:ho']

# TeV refinement
from RecoMuon.GlobalMuonProducer.tevMuons_cfi import *
tevMuons.MuonCollectionLabel = 'globalCosmicMuons'
tevMuons.RefitterParameters.PropDirForCosmics = True

# Glb Track Quality
from RecoMuon.GlobalTrackingTools.GlobalTrackQuality_cfi import *
glbTrackQual.InputCollection = 'globalCosmicMuons'

# all muons id
allmuonsTask = cms.Task(glbTrackQual,
                        tevMuons,
                        muons,
                        muIsolationTask)
allmuons = cms.Sequence(allmuonsTask)

# Final sequence
muonrecoforcosmicsTask = cms.Task(muontrackingforcosmicsTask,
				  allmuonsTask,
                                  muonsFromCosmics)
muonrecoforcosmics = cms.Sequence(muonrecoforcosmicsTask)

# 1 leg mode
# Stand alone muon track producer
cosmicMuons1Leg = cosmicMuons.clone(
    MuonSeedCollectionLabel = 'CosmicMuonSeed',
    TrajectoryBuilderParameters = dict(BuildTraversingMuon = True)
)

# Global muon track producer
globalCosmicMuons1Leg = globalCosmicMuons.clone(
    MuonCollectionLabel = 'cosmicMuons1Leg',
    TrajectoryBuilderParameters = dict(TkTrackCollectionLabel = 'ctfWithMaterialTracksP5')
)

# Muon Id producer
muons1Leg = muons1stStep.clone(
    inputCollectionLabels = ['ctfWithMaterialTracksP5', 
                             'globalCosmicMuons1Leg', 
                             'cosmicMuons1Leg'],
    inputCollectionTypes = ['inner tracks', 'links', 'outer tracks'],
    fillIsolation = False,
    fillGlobalTrackQuality = False,
    fillGlobalTrackRefits = False,
    TimingFillerParameters = dict(
        DTTimingParameters = dict(PruneCut = 9999),
        CSCTimingParameters = dict(PruneCut = 9999))
)

# Sequences
# Stand Alone Tracking
STAmuontrackingforcosmics1LegTask = cms.Task(CosmicMuonSeed,cosmicMuons1Leg)

# Stand Alone Tracking plus global tracking
muontrackingforcosmics1LegTask = cms.Task(STAmuontrackingforcosmics1LegTask, globalCosmicMuons1Leg)

# Final sequence
muonrecoforcosmics1LegTask = cms.Task(muontrackingforcosmics1LegTask,muons1Leg)

#####################################################

# t0 Corrections

# Seed generator
CosmicMuonSeedWitht0Correction = CosmicMuonSeed.clone(
    DTRecSegmentLabel = 'dt4DSegmentsT0Seg'
)

# Stand alone muon track producer
cosmicMuonsWitht0Correction = cosmicMuons.clone(
    MuonSeedCollectionLabel = 'CosmicMuonSeedWitht0Correction',
    TrajectoryBuilderParameters = dict(BuildTraversingMuon = False, DTRecSegmentLabel = 'dt4DSegmentsT0Seg')
)

# Global muon track producer
globalCosmicMuonsWitht0Correction = globalCosmicMuons.clone(
    MuonCollectionLabel = 'cosmicMuonsWitht0Correction',
    TrajectoryBuilderParameters = dict(TkTrackCollectionLabel = 'ctfWithMaterialTracksP5')
)

# Muon Id producer
muonsWitht0Correction = muons1stStep.clone(
    inputCollectionLabels = ['ctfWithMaterialTracksP5', 
                             'globalCosmicMuonsWitht0Correction', 
                             'cosmicMuonsWitht0Correction'],
    inputCollectionTypes = ['inner tracks', 'links', 'outer tracks'],
    fillIsolation = True,
    fillGlobalTrackQuality = False,
    fillGlobalTrackRefits = False,
    TrackExtractorPSet = dict(inputTrackCollection = 'ctfWithMaterialTracksP5'),
    # set wide cone until the code is made to compute this wrt CalIntersection
    CaloExtractorPSet = dict(CenterConeOnCalIntersection = True, DR_Max = 1.0),
    TimingFillerParameters = dict(
	DTTimingParameters = dict(UseSegmentT0 = True),
	MatchParameters = dict(DTsegments = 'dt4DSegmentsT0Seg'))
)

#Sequences
# Stand Alone Tracking
STAmuontrackingforcosmicsWitht0CorrectionTask = cms.Task(CosmicMuonSeedWitht0Correction,cosmicMuonsWitht0Correction)
STAmuontrackingforcosmicsWitht0Correction = cms.Sequence(STAmuontrackingforcosmicsWitht0CorrectionTask)

# Stand Alone Tracking plus global tracking
muontrackingforcosmicsWitht0CorrectionTask = cms.Task(STAmuontrackingforcosmicsWitht0CorrectionTask,globalCosmicMuonsWitht0Correction)
muontrackingforcosmicsWitht0Correction = cms.Sequence(muontrackingforcosmicsWitht0CorrectionTask)

# Stand Alone Tracking plus muon ID
STAmuonrecoforcosmicsWitht0Correction = cms.Sequence(STAmuontrackingforcosmicsWitht0CorrectionTask)

# Final sequence
muonrecoforcosmicsWitht0CorrectionTask = cms.Task(muontrackingforcosmicsWitht0CorrectionTask,muonsWitht0Correction)
muonrecoforcosmicsWitht0Correction = cms.Sequence(muonrecoforcosmicsWitht0CorrectionTask)

### Final sequence ###
muonRecoGRTask = cms.Task(muonrecoforcosmics1LegTask,muonrecoforcosmicsWitht0CorrectionTask)
muonRecoGR = cms.Sequence(muonRecoGRTask)

#####################################################

# Beam halo in Encaps only. GLB reco only is needed

# Seed generator 
CosmicMuonSeedEndCapsOnly = CosmicMuonSeed.clone(
    EnableDTMeasurement = False
)

# Stand alone muon track producer
cosmicMuonsEndCapsOnly = cosmicMuons.clone(
    MuonSeedCollectionLabel = 'CosmicMuonSeedEndCapsOnly',
    TrajectoryBuilderParameters = dict(
	EnableDTMeasurement = False,
	MuonNavigationParameters = dict(Barrel = False))
)

# Global muon track producer
globalBeamHaloMuonEndCapslOnly = globalCosmicMuons.clone(
    MuonCollectionLabel = 'cosmicMuonsEndCapsOnly'
)

# Muon Id producer
muonsBeamHaloEndCapsOnly = muons1stStep.clone(
    inputCollectionLabels = ['beamhaloTracks', 
                             'globalBeamHaloMuonEndCapslOnly', 
                             'cosmicMuonsEndCapsOnly'],
    inputCollectionTypes = ['inner tracks', 'links', 'outer tracks'],
    fillIsolation = True,
    fillGlobalTrackQuality = False,
    fillGlobalTrackRefits = False,
    TrackExtractorPSet = dict(inputTrackCollection = 'ctfWithMaterialTracksP5'),
    # set wide cone until the code is made to compute this wrt CalIntersection
    CaloExtractorPSet = dict(CenterConeOnCalIntersection = True, DR_Max = 1.0)
)

# Sequences
muonrecoBeamHaloEndCapsOnlyTask = cms.Task(CosmicMuonSeedEndCapsOnly,
                                           cosmicMuonsEndCapsOnly,
                                           globalBeamHaloMuonEndCapslOnly,
                                           muonsBeamHaloEndCapsOnly)
muonrecoBeamHaloEndCapsOnly = cms.Sequence(muonrecoBeamHaloEndCapsOnlyTask)

########

## Full detector but NO RPC ##

# Stand alone muon track producer
cosmicMuonsNoRPC = cosmicMuons.clone(
    TrajectoryBuilderParameters = dict(EnableRPCMeasurement = False)
)

# Global muon track producer
globalCosmicMuonsNoRPC = globalCosmicMuons.clone(
    MuonCollectionLabel = 'cosmicMuonsNoRPC',
    TrajectoryBuilderParameters = dict(TkTrackCollectionLabel = 'ctfWithMaterialTracksP5')
)

# Muon Id producer
muonsNoRPC = muons1stStep.clone(
    inputCollectionLabels = ['ctfWithMaterialTracksP5',
                             'globalCosmicMuonsNoRPC', 
                             'cosmicMuonsNoRPC'],
    inputCollectionTypes = ['inner tracks', 'links', 'outer tracks'],
    fillIsolation = True,
    fillGlobalTrackQuality = False,
    fillGlobalTrackRefits = False,
    TrackExtractorPSet = dict(inputTrackCollection = 'ctfWithMaterialTracksP5'),
    # set wide cone until the code is made to compute this wrt CalIntersection
    CaloExtractorPSet = dict(CenterConeOnCalIntersection = True, DR_Max = 1.0)
)

#Sequences

# Stand Alone Tracking plus global tracking
muontrackingforcosmicsNoRPCTask = cms.Task(cosmicMuonsNoRPC,globalCosmicMuonsNoRPC)
muontrackingforcosmicsNoRPC = cms.Sequence(muontrackingforcosmicsNoRPCTask)

# Final sequence
muonrecoforcosmicsNoRPCTask = cms.Task(muontrackingforcosmicsNoRPCTask,muonsNoRPC)
muonrecoforcosmicsNoRPC = cms.Sequence(muonrecoforcosmicsNoRPCTask)

##############################################

## Split Tracks  ##

# Global muon track producer
globalCosmicSplitMuons = globalCosmicMuons.clone(
    MuonCollectionLabel = 'cosmicMuons',
    TrajectoryBuilderParameters = dict(TkTrackCollectionLabel = 'splittedTracksP5')
)

# Muon Id producer
splitMuons = muons1stStep.clone(
    inputCollectionLabels = ['splittedTracksP5', 
                             'globalCosmicSplitMuons', 
                             'cosmicMuons'],
    inputCollectionTypes = ['inner tracks', 'links', 'outer tracks'],
    fillIsolation = True,
    fillGlobalTrackQuality = False,
    fillGlobalTrackRefits = False,
    TrackExtractorPSet = dict(inputTrackCollection = 'splittedTracksP5'),
    # set wide cone until the code is made to compute this wrt CalIntersection
    CaloExtractorPSet = dict(CenterConeOnCalIntersection = True, DR_Max = 1.0)
)

#Sequences

# Final sequence
muonrecoforsplitcosmicsTask = cms.Task(globalCosmicSplitMuons,splitMuons)
muonrecoforsplitcosmics = cms.Sequence(muonrecoforsplitcosmicsTask)

##############################################

######################## LHC like Reco #############################

# Standard reco
from RecoMuon.Configuration.RecoMuonPPonly_cff import *

# Muon Id producer
lhcSTAMuons = muons1stStep.clone(
    inputCollectionLabels = ['standAloneMuons'],
    inputCollectionTypes = ['outer tracks'],
    fillIsolation = True,
    fillGlobalTrackQuality = False,
    fillGlobalTrackRefits = False,
    TrackExtractorPSet = dict(inputTrackCollection = 'ctfWithMaterialTracksP5LHCNavigation'),
    # set wide cone until the code is made to compute this wrt CalIntersection
    CaloExtractorPSet = dict(CenterConeOnCalIntersection = True, DR_Max = 1.0)
)

# Final sequence
muonRecoLHCTask = cms.Task(ancientMuonSeed,
                           standAloneMuons,
                           lhcSTAMuons)
muonRecoLHC = cms.Sequence(muonRecoLHCTask)

########################### SEQUENCE TO BE ADDED in ReconstructionGR_cff ##############################################
muonRecoGRTask = cms.Task(muonrecoforcosmicsTask,
                          muonRecoGRTask,
                          muonrecoBeamHaloEndCapsOnlyTask,
                          muonrecoforcosmicsNoRPCTask,
                          muonrecoforsplitcosmicsTask,
                          muonRecoLHCTask)
muonRecoGR = cms.Sequence(muonRecoGRTask)
#######################################################################################################################
