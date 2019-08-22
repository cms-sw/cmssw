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

muons = muons1stStep.clone()

muons.inputCollectionLabels = ['ctfWithMaterialTracksP5LHCNavigation', 'globalCosmicMuons', 'cosmicMuons', "tevMuons:firstHit","tevMuons:picky","tevMuons:dyt"]
muons.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks', 'tev firstHit', 'tev picky', 'tev dyt']
muons.fillIsolation = True
muons.fillGlobalTrackQuality = True
muons.TimingFillerParameters.DTTimingParameters.PruneCut = 9999
muons.TimingFillerParameters.CSCTimingParameters.PruneCut = 9999
# need to modify track selection as well (not clear to what)
muons.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5LHCNavigation'
muons.CaloExtractorPSet.CenterConeOnCalIntersection = True
# set wide cone until the code is made to compute this wrt CalIntersection
muons.CaloExtractorPSet.DR_Max = 1.0

#similar to what's in pp configuration
muonsFromCosmics = muons1stStep.clone()
muonsFromCosmics.inputCollectionLabels = ['cosmicMuons']
muonsFromCosmics.inputCollectionTypes = ['outer tracks']
muonsFromCosmics.TrackExtractorPSet.inputTrackCollection = 'cosmicMuons'
muonsFromCosmics.TimingFillerParameters.DTTimingParameters.PruneCut = 9999
muonsFromCosmics.TimingFillerParameters.CSCTimingParameters.PruneCut = 9999
muonsFromCosmics.fillIsolation = False
muonsFromCosmics.fillGlobalTrackQuality = False
muonsFromCosmics.fillGlobalTrackRefits = False

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
muIsoDepositTk.inputTags = cms.VInputTag(cms.InputTag("muons:tracker"))
muIsoDepositJets. inputTags = cms.VInputTag(cms.InputTag("muons:jets"))
muIsoDepositCalByAssociatorTowers.inputTags = cms.VInputTag(cms.InputTag("muons:ecal"), cms.InputTag("muons:hcal"), cms.InputTag("muons:ho"))



# TeV refinement
from RecoMuon.GlobalMuonProducer.tevMuons_cfi import *
tevMuons.MuonCollectionLabel = "globalCosmicMuons"
tevMuons.RefitterParameters.PropDirForCosmics = cms.bool(True)

# Glb Track Quality
from RecoMuon.GlobalTrackingTools.GlobalTrackQuality_cfi import *
glbTrackQual.InputCollection = "globalCosmicMuons"

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
cosmicMuons1Leg = cosmicMuons.clone()
cosmicMuons1Leg.TrajectoryBuilderParameters.BuildTraversingMuon = True
cosmicMuons1Leg.MuonSeedCollectionLabel = 'CosmicMuonSeed'

# Global muon track producer
globalCosmicMuons1Leg = globalCosmicMuons.clone()
globalCosmicMuons1Leg.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'
globalCosmicMuons1Leg.MuonCollectionLabel = 'cosmicMuons1Leg'

# Muon Id producer
muons1Leg = muons1stStep.clone()
muons1Leg.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuons1Leg', 'cosmicMuons1Leg']
muons1Leg.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muons1Leg.fillIsolation = False
muons1Leg.fillGlobalTrackQuality = False
muons1Leg.fillGlobalTrackRefits = False
muons1Leg.TimingFillerParameters.DTTimingParameters.PruneCut = 9999
muons1Leg.TimingFillerParameters.CSCTimingParameters.PruneCut = 9999
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
CosmicMuonSeedWitht0Correction = CosmicMuonSeed.clone()
CosmicMuonSeedWitht0Correction.DTRecSegmentLabel = 'dt4DSegmentsT0Seg'

# Stand alone muon track producer
cosmicMuonsWitht0Correction = cosmicMuons.clone()
cosmicMuonsWitht0Correction.TrajectoryBuilderParameters.BuildTraversingMuon = False
cosmicMuonsWitht0Correction.MuonSeedCollectionLabel = 'CosmicMuonSeedWitht0Correction'
cosmicMuonsWitht0Correction.TrajectoryBuilderParameters.DTRecSegmentLabel = 'dt4DSegmentsT0Seg'

# Global muon track producer
globalCosmicMuonsWitht0Correction = globalCosmicMuons.clone()
globalCosmicMuonsWitht0Correction.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'
globalCosmicMuonsWitht0Correction.MuonCollectionLabel = 'cosmicMuonsWitht0Correction'

# Muon Id producer
muonsWitht0Correction = muons1stStep.clone()
muonsWitht0Correction.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuonsWitht0Correction', 'cosmicMuonsWitht0Correction']
muonsWitht0Correction.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsWitht0Correction.fillIsolation = True
muonsWitht0Correction.fillGlobalTrackQuality = False
muonsWitht0Correction.TimingFillerParameters.DTTimingParameters.UseSegmentT0 = True
muonsWitht0Correction.TimingFillerParameters.MatchParameters.DTsegments = 'dt4DSegmentsT0Seg'
muonsWitht0Correction.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5'
muonsWitht0Correction.CaloExtractorPSet.CenterConeOnCalIntersection = True
# set wide cone until the code is made to compute this wrt CalIntersection
muonsWitht0Correction.CaloExtractorPSet.DR_Max = 1.0
muonsWitht0Correction.fillGlobalTrackRefits = False
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
CosmicMuonSeedEndCapsOnly = CosmicMuonSeed.clone()
CosmicMuonSeedEndCapsOnly.EnableDTMeasurement = False

# Stand alone muon track producer
cosmicMuonsEndCapsOnly = cosmicMuons.clone()
cosmicMuonsEndCapsOnly.TrajectoryBuilderParameters.EnableDTMeasurement = False
cosmicMuonsEndCapsOnly.TrajectoryBuilderParameters.MuonNavigationParameters.Barrel = False
cosmicMuonsEndCapsOnly.MuonSeedCollectionLabel = 'CosmicMuonSeedEndCapsOnly'

# Global muon track producer
globalBeamHaloMuonEndCapslOnly = globalCosmicMuons.clone()
globalBeamHaloMuonEndCapslOnly.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'beamhaloTracks'
globalBeamHaloMuonEndCapslOnly.MuonCollectionLabel = 'cosmicMuonsEndCapsOnly'


# Muon Id producer
muonsBeamHaloEndCapsOnly = muons1stStep.clone()           
muonsBeamHaloEndCapsOnly.inputCollectionLabels = ['beamhaloTracks', 'globalBeamHaloMuonEndCapslOnly', 'cosmicMuonsEndCapsOnly']
muonsBeamHaloEndCapsOnly.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsBeamHaloEndCapsOnly.fillIsolation = True
muonsBeamHaloEndCapsOnly.fillGlobalTrackQuality = False
muonsBeamHaloEndCapsOnly.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5'
muonsBeamHaloEndCapsOnly.CaloExtractorPSet.CenterConeOnCalIntersection = True
# set wide cone until the code is made to compute this wrt CalIntersection
muonsBeamHaloEndCapsOnly.CaloExtractorPSet.DR_Max = 1.0
muonsBeamHaloEndCapsOnly.fillGlobalTrackRefits = False

# Sequences
muonrecoBeamHaloEndCapsOnlyTask = cms.Task(CosmicMuonSeedEndCapsOnly,
                                           cosmicMuonsEndCapsOnly,
                                           globalBeamHaloMuonEndCapslOnly,
                                           muonsBeamHaloEndCapsOnly)
muonrecoBeamHaloEndCapsOnly = cms.Sequence(muonrecoBeamHaloEndCapsOnlyTask)

########

## Full detector but NO RPC ##

# Stand alone muon track producer
cosmicMuonsNoRPC = cosmicMuons.clone()
cosmicMuonsNoRPC.TrajectoryBuilderParameters.EnableRPCMeasurement = False

# Global muon track producer
globalCosmicMuonsNoRPC = globalCosmicMuons.clone()
globalCosmicMuonsNoRPC.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'
globalCosmicMuonsNoRPC.MuonCollectionLabel = 'cosmicMuonsNoRPC'

# Muon Id producer
muonsNoRPC = muons1stStep.clone()
muonsNoRPC.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuonsNoRPC', 'cosmicMuonsNoRPC']
muonsNoRPC.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsNoRPC.fillIsolation = True
muonsNoRPC.fillGlobalTrackQuality = False
muonsNoRPC.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5'
muonsNoRPC.CaloExtractorPSet.CenterConeOnCalIntersection = True
# set wide cone until the code is made to compute this wrt CalIntersection
muonsNoRPC.CaloExtractorPSet.DR_Max = 1.0
muonsNoRPC.fillGlobalTrackRefits = False

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
globalCosmicSplitMuons = globalCosmicMuons.clone()
globalCosmicSplitMuons.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'splittedTracksP5'
globalCosmicSplitMuons.MuonCollectionLabel = 'cosmicMuons'

# Muon Id producer

splitMuons = muons1stStep.clone()
splitMuons.inputCollectionLabels = ['splittedTracksP5', 'globalCosmicSplitMuons', 'cosmicMuons']
splitMuons.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
splitMuons.fillIsolation = True
splitMuons.fillGlobalTrackQuality = False
splitMuons.TrackExtractorPSet.inputTrackCollection = 'splittedTracksP5'
splitMuons.CaloExtractorPSet.CenterConeOnCalIntersection = True
# set wide cone until the code is made to compute this wrt CalIntersection
splitMuons.CaloExtractorPSet.DR_Max = 1.0
splitMuons.fillGlobalTrackRefits = False

#Sequences

# Final sequence
muonrecoforsplitcosmicsTask = cms.Task(globalCosmicSplitMuons,splitMuons)
muonrecoforsplitcosmics = cms.Sequence(muonrecoforsplitcosmicsTask)

##############################################

######################## LHC like Reco #############################

# Standard reco
from RecoMuon.Configuration.RecoMuonPPonly_cff import *

# Muon Id producer
lhcSTAMuons = muons1stStep.clone()
lhcSTAMuons.inputCollectionLabels = ['standAloneMuons']
lhcSTAMuons.inputCollectionTypes = ['outer tracks']
lhcSTAMuons.fillIsolation = True
lhcSTAMuons.fillGlobalTrackQuality = False
lhcSTAMuons.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5LHCNavigation'
lhcSTAMuons.CaloExtractorPSet.CenterConeOnCalIntersection = True
# set wide cone until the code is made to compute this wrt CalIntersection
lhcSTAMuons.CaloExtractorPSet.DR_Max = 1.0
lhcSTAMuons.fillGlobalTrackRefits = False

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





