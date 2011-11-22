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
# need to modify track selection as well (not clear to what)
muons.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5LHCNavigation'
muons.CaloExtractorPSet.CenterConeOnCalIntersection = True

from RecoMuon.MuonIdentification.calomuons_cfi import *
calomuons.inputTracks = 'ctfWithMaterialTracksP5LHCNavigation'
calomuons.inputCollection = 'muons'
calomuons.inputMuons = 'muons'

## Sequences

# Stand Alone Tracking
STAmuontrackingforcosmics = cms.Sequence(CosmicMuonSeed*cosmicMuons)
# Stand Alone Tracking plus muon ID
STAmuonrecoforcosmics = cms.Sequence(STAmuontrackingforcosmics)

# Stand Alone Tracking plus global tracking
muontrackingforcosmics = cms.Sequence(STAmuontrackingforcosmics*globalCosmicMuons)

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
allmuons = cms.Sequence(glbTrackQual*tevMuons*muons*muIsolation*calomuons)

# Final sequence
muonrecoforcosmics = cms.Sequence(muontrackingforcosmics*allmuons)
muonRecoAllGR = cms.Sequence(muonrecoforcosmics)

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
# Sequences

# Stand Alone Tracking
STAmuontrackingforcosmics1Leg = cms.Sequence(CosmicMuonSeed*cosmicMuons1Leg)

# Stand Alone Tracking plus global tracking
muontrackingforcosmics1Leg = cms.Sequence(STAmuontrackingforcosmics1Leg*globalCosmicMuons1Leg)

# all muons id
allmuons1Leg = cms.Sequence(muons1Leg)

# Stand Alone Tracking plus muon ID
STAmuonrecoforcosmics1Leg = cms.Sequence(STAmuontrackingforcosmics1Leg)

# Final sequence
muonrecoforcosmics1Leg = cms.Sequence(muontrackingforcosmics1Leg*allmuons1Leg)

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
muonsWitht0Correction.TimingFillerParameters.DTTimingParameters.DTsegments = 'dt4DSegmentsT0Seg'
muonsWitht0Correction.TimingFillerParameters.DTTimingParameters.MatchParameters.DTsegments = 'dt4DSegmentsT0Seg'
muonsWitht0Correction.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5'
muonsWitht0Correction.CaloExtractorPSet.CenterConeOnCalIntersection = True
muonsWitht0Correction.fillGlobalTrackRefits = False
#Sequences


# Stand Alone Tracking
STAmuontrackingforcosmicsWitht0Correction = cms.Sequence(CosmicMuonSeedWitht0Correction*cosmicMuonsWitht0Correction)

# Stand Alone Tracking plus global tracking
muontrackingforcosmicsWitht0Correction = cms.Sequence(STAmuontrackingforcosmicsWitht0Correction*globalCosmicMuonsWitht0Correction)

# Stand Alone Tracking plus muon ID
STAmuonrecoforcosmicsWitht0Correction = cms.Sequence(STAmuontrackingforcosmicsWitht0Correction)

# all muons id
allmuonsWitht0Correction = cms.Sequence(muonsWitht0Correction)

# Final sequence
muonrecoforcosmicsWitht0Correction = cms.Sequence(muontrackingforcosmicsWitht0Correction*allmuonsWitht0Correction)

### Final sequence ###
muonRecoGR = cms.Sequence(muonrecoforcosmics1Leg+muonrecoforcosmicsWitht0Correction)

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
muonsBeamHaloEndCapsOnly.fillGlobalTrackRefits = False

# Sequences
muonrecoBeamHaloEndCapsOnly = cms.Sequence(CosmicMuonSeedEndCapsOnly*cosmicMuonsEndCapsOnly*globalBeamHaloMuonEndCapslOnly*muonsBeamHaloEndCapsOnly)

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
muonsNoRPC.fillGlobalTrackRefits = False

#Sequences

# Stand Alone Tracking
STAmuontrackingforcosmicsNoRPC = cms.Sequence(cosmicMuonsNoRPC)

# Stand Alone Tracking plus global tracking
muontrackingforcosmicsNoRPC = cms.Sequence(STAmuontrackingforcosmicsNoRPC*globalCosmicMuonsNoRPC)

# all muons id
allmuonsNoRPC = cms.Sequence(muonsNoRPC)

# Final sequence
muonrecoforcosmicsNoRPC = cms.Sequence(muontrackingforcosmicsNoRPC*allmuonsNoRPC)

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
splitMuons.fillGlobalTrackRefits = False

#Sequences

# Final sequence
muonrecoforsplitcosmics = cms.Sequence(globalCosmicSplitMuons*splitMuons)

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
lhcSTAMuons.fillGlobalTrackRefits = False

# Final sequence
muonRecoLHC = cms.Sequence(ancientMuonSeed*standAloneMuons*lhcSTAMuons)



########################### SEQUENCE TO BE ADDED in ReconstructionGR_cff ##############################################

muonRecoGR = cms.Sequence(muonRecoAllGR*muonRecoGR*muonrecoBeamHaloEndCapsOnly*muonrecoforcosmicsNoRPC*muonrecoforsplitcosmics*muonRecoLHC)

#######################################################################################################################





