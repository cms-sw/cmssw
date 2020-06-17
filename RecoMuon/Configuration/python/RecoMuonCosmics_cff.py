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
    TimingFillerParameters.DTTimingParameters.PruneCut = 9999,
    TimingFillerParameters.CSCTimingParameters.PruneCut = 9999,
    # need to modify track selection as well (not clear to what)
    TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5LHCNavigation',
    CaloExtractorPSet.CenterConeOnCalIntersection = True,
    # set wide cone until the code is made to compute this wrt CalIntersection
    CaloExtractorPSet.DR_Max = 1.0
)

#similar to what's in pp configuration
muonsFromCosmics = muons1stStep.clone(
    inputCollectionLabels = ['cosmicMuons'],
    inputCollectionTypes = ['outer tracks'],
    TrackExtractorPSet.inputTrackCollection = 'cosmicMuons',
    TimingFillerParameters.DTTimingParameters.PruneCut = 9999,
    TimingFillerParameters.CSCTimingParameters.PruneCut = 9999,
    fillIsolation = False,
    fillGlobalTrackQuality = False,
    fillGlobalTrackRefits = False
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
muIsoDepositTk.inputTags = 'muons:tracker'
muIsoDepositJets.inputTags = 'muons:jets'
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
    TrajectoryBuilderParameters.BuildTraversingMuon = True,
    MuonSeedCollectionLabel = 'CosmicMuonSeed'
)

# Global muon track producer
globalCosmicMuons1Leg = globalCosmicMuons.clone(
    TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5',
    MuonCollectionLabel = 'cosmicMuons1Leg'
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
    TimingFillerParameters.DTTimingParameters.PruneCut = 9999,
    TimingFillerParameters.CSCTimingParameters.PruneCut = 9999
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
    TrajectoryBuilderParameters.BuildTraversingMuon = False,
    MuonSeedCollectionLabel = 'CosmicMuonSeedWitht0Correction',
    TrajectoryBuilderParameters.DTRecSegmentLabel = 'dt4DSegmentsT0Seg'
)

# Global muon track producer
globalCosmicMuonsWitht0Correction = globalCosmicMuons.clone(
    TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5',
    MuonCollectionLabel = 'cosmicMuonsWitht0Correction'
)

# Muon Id producer
muonsWitht0Correction = muons1stStep.clone(
    inputCollectionLabels = ['ctfWithMaterialTracksP5', 
                             'globalCosmicMuonsWitht0Correction', 
                             'cosmicMuonsWitht0Correction'],
    inputCollectionTypes = ['inner tracks', 'links', 'outer tracks'],
    fillIsolation = True,
    fillGlobalTrackQuality = False,
    TimingFillerParameters.DTTimingParameters.UseSegmentT0 = True,
    TimingFillerParameters.MatchParameters.DTsegments = 'dt4DSegmentsT0Seg',
    TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5',
    CaloExtractorPSet.CenterConeOnCalIntersection = True,
    # set wide cone until the code is made to compute this wrt CalIntersection
    CaloExtractorPSet.DR_Max = 1.0,
    fillGlobalTrackRefits = False
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
    TrajectoryBuilderParameters.EnableDTMeasurement = False,
    TrajectoryBuilderParameters.MuonNavigationParameters.Barrel = False,
    MuonSeedCollectionLabel = 'CosmicMuonSeedEndCapsOnly'
)

# Global muon track producer
globalBeamHaloMuonEndCapslOnly = globalCosmicMuons.clone(
    TrajectoryBuilderParameters.TkTrackCollectionLabel = 'beamhaloTracks',
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
    TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5',
    CaloExtractorPSet.CenterConeOnCalIntersection = True,
    # set wide cone until the code is made to compute this wrt CalIntersection
    CaloExtractorPSet.DR_Max = 1.0,
    fillGlobalTrackRefits = False
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
    TrajectoryBuilderParameters.EnableRPCMeasurement = False
)
# Global muon track producer
globalCosmicMuonsNoRPC = globalCosmicMuons.clone(
    TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5',
    MuonCollectionLabel = 'cosmicMuonsNoRPC'
)

# Muon Id producer
muonsNoRPC = muons1stStep.clone(
    inputCollectionLabels = ['ctfWithMaterialTracksP5',
                             'globalCosmicMuonsNoRPC', 
                             'cosmicMuonsNoRPC'],
    inputCollectionTypes = ['inner tracks', 'links', 'outer tracks'],
    fillIsolation = True,
    fillGlobalTrackQuality = False,
    TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5',
    CaloExtractorPSet.CenterConeOnCalIntersection = True,
    # set wide cone until the code is made to compute this wrt CalIntersection
    CaloExtractorPSet.DR_Max = 1.0,
    fillGlobalTrackRefits = False
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
    TrajectoryBuilderParameters.TkTrackCollectionLabel = 'splittedTracksP5',
    MuonCollectionLabel = 'cosmicMuons'
)

# Muon Id producer
splitMuons = muons1stStep.clone(
    inputCollectionLabels = ['splittedTracksP5', 
                             'globalCosmicSplitMuons', 
                             'cosmicMuons'],
    inputCollectionTypes = ['inner tracks', 'links', 'outer tracks'],
    fillIsolation = True,
    fillGlobalTrackQuality = False,
    TrackExtractorPSet.inputTrackCollection = 'splittedTracksP5',
    CaloExtractorPSet.CenterConeOnCalIntersection = True,
    # set wide cone until the code is made to compute this wrt CalIntersection
    CaloExtractorPSet.DR_Max = 1.0,
    fillGlobalTrackRefits = False
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
    TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5LHCNavigation',
    CaloExtractorPSet.CenterConeOnCalIntersection = True,
    # set wide cone until the code is made to compute this wrt CalIntersection
    CaloExtractorPSet.DR_Max = 1.0,
    fillGlobalTrackRefits = False
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
