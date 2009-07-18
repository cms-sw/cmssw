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

muons.inputCollectionLabels = ['ctfWithMaterialTracksP5LHCNavigation', 'globalCosmicMuons', 'cosmicMuons']
muons.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muons.fillIsolation = True
# need to modify track selection as well (not clear to what)
muons.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5LHCNavigation'
muons.CaloExtractorPSet.CenterConeOnCalIntersection = True

from RecoMuon.MuonIdentification.calomuons_cfi import *
calomuons.inputTracks = 'ctfWithMaterialTracksP5LHCNavigation'

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
muIsoDepositTk.ExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5LHCNavigation'
muIsoDepositCalByAssociatorTowers.ExtractorPSet.CenterConeOnCalIntersection = True

# TeV refinement
from RecoMuon.GlobalMuonProducer.tevMuons_cfi import *
tevMuons.MuonCollectionLabel = "globalCosmicMuons"

# all muons id
allmuons = cms.Sequence(muons*tevMuons*muIsolation*calomuons)

# Final sequence
muonrecoforcosmics = cms.Sequence(muontrackingforcosmics*allmuons)
muonRecoAllGR = cms.Sequence(muonrecoforcosmics)

##############################################

## Barrel only ##

# Seed generator 
CosmicMuonSeedBarrelOnly = CosmicMuonSeed.clone()
CosmicMuonSeedBarrelOnly.EnableCSCMeasurement = False

# Stand alone muon track producer
cosmicMuonsBarrelOnly = cosmicMuons.clone()
cosmicMuonsBarrelOnly.TrajectoryBuilderParameters.EnableCSCMeasurement = False
cosmicMuonsBarrelOnly.TrajectoryBuilderParameters.MuonNavigationParameters.Endcap = False
cosmicMuonsBarrelOnly.MuonSeedCollectionLabel = 'CosmicMuonSeedBarrelOnly'

# Global muon track producer
globalCosmicMuonsBarrelOnly = globalCosmicMuons.clone()
globalCosmicMuonsBarrelOnly.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'
globalCosmicMuonsBarrelOnly.MuonCollectionLabel = 'cosmicMuonsBarrelOnly'

# Muon Id producer
muonsBarrelOnly = muons.clone()
muonsBarrelOnly.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuonsBarrelOnly', 'cosmicMuonsBarrelOnly']
muonsBarrelOnly.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsBarrelOnly.fillIsolation = True
muonsBarrelOnly.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5'
muonsBarrelOnly.CaloExtractorPSet.CenterConeOnCalIntersection = True

#Sequences

# Stand Alone Tracking
STAmuontrackingforcosmicsBarrelOnly = cms.Sequence(CosmicMuonSeedBarrelOnly*cosmicMuonsBarrelOnly)

# Stand Alone Tracking plus global tracking
muontrackingforcosmicsBarrelOnly = cms.Sequence(STAmuontrackingforcosmicsBarrelOnly*globalCosmicMuonsBarrelOnly)

# Stand Alone Tracking plus muon ID
STAmuonrecoforcosmicsBarrelOnly = cms.Sequence(STAmuontrackingforcosmicsBarrelOnly)

# all muons id
allmuonsBarrelOnly = cms.Sequence(muonsBarrelOnly)

# Final sequence
muonrecoforcosmicsBarrelOnly = cms.Sequence(muontrackingforcosmicsBarrelOnly*allmuonsBarrelOnly)

########

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
muons1Leg = muons.clone()
muons1Leg.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuons1Leg', 'cosmicMuons1Leg']
muons1Leg.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muons1Leg.fillIsolation = False

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

########

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
muonsWitht0Correction = muons.clone()
muonsWitht0Correction.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuonsWitht0Correction', 'cosmicMuonsWitht0Correction']
muonsWitht0Correction.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsWitht0Correction.fillIsolation = True
muonsWitht0Correction.TimingFillerParameters.DTTimingParameters.UseSegmentT0 = True
muonsWitht0Correction.TimingFillerParameters.DTTimingParameters.DTsegments = 'dt4DSegmentsT0Seg'
muonsWitht0Correction.TimingFillerParameters.DTTimingParameters.MatchParameters.DTsegments = 'dt4DSegmentsT0Seg'
muonsWitht0Correction.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5'
muonsWitht0Correction.CaloExtractorPSet.CenterConeOnCalIntersection = True

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

### Final sequence for barrel only ###
muonRecoBarrelGR = cms.Sequence(muonrecoforcosmicsBarrelOnly+muonrecoforcosmics1Leg+muonrecoforcosmicsWitht0Correction)

##############################################

## Endcaps only ##

# Seed generator 
CosmicMuonSeedEndCapsOnly = CosmicMuonSeed.clone()
CosmicMuonSeedEndCapsOnly.EnableDTMeasurement = False

# Stand alone muon track producer
cosmicMuonsEndCapsOnly = cosmicMuons.clone()
cosmicMuonsEndCapsOnly.TrajectoryBuilderParameters.EnableDTMeasurement = False
cosmicMuonsEndCapsOnly.TrajectoryBuilderParameters.MuonNavigationParameters.Barrel = False
cosmicMuonsEndCapsOnly.MuonSeedCollectionLabel = 'CosmicMuonSeedEndCapsOnly'

# Global muon track producer
globalCosmicMuonsEndCapsOnly = globalCosmicMuons.clone()
globalCosmicMuonsEndCapsOnly.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'
globalCosmicMuonsEndCapsOnly.MuonCollectionLabel = 'cosmicMuonsEndCapsOnly'

# Muon Id producer
muonsEndCapsOnly = muons.clone()
muonsEndCapsOnly.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuonsEndCapsOnly', 'cosmicMuonsEndCapsOnly']
muonsEndCapsOnly.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsEndCapsOnly.fillIsolation = True
muonsEndCapsOnly.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5'
muonsEndCapsOnly.CaloExtractorPSet.CenterConeOnCalIntersection = True

# Sequences

# Stand Alone Tracking
STAmuontrackingforcosmicsEnsCapsOnly = cms.Sequence(CosmicMuonSeedEndCapsOnly*cosmicMuonsEndCapsOnly)

# Stand Alone Tracking plus global tracking
muontrackingforcosmicsEndCapsOnly = cms.Sequence(STAmuontrackingforcosmicsEnsCapsOnly*globalCosmicMuonsEndCapsOnly)

# Stand Alone Tracking plus muon ID
STAmuonrecoforcosmicsEndCapsOnly = cms.Sequence(STAmuontrackingforcosmicsEnsCapsOnly)

# all muons id
allmuonsEndCapsOnly = cms.Sequence(muonsEndCapsOnly)

# Final sequence
muonrecoforcosmicsEndCapsOnly = cms.Sequence(muontrackingforcosmicsEndCapsOnly*allmuonsEndCapsOnly)

########

# Beam halo in Encaps only. GLB reco only is needed
globalBeamHaloMuonEndCapslOnly = globalCosmicMuonsEndCapsOnly.clone()
globalBeamHaloMuonEndCapslOnly.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksBeamHaloMuon'

# Muon Id producer
muonsBeamHaloEndCapsOnly = muons.clone()           
muonsBeamHaloEndCapsOnly.inputCollectionLabels = ['ctfWithMaterialTracksBeamHaloMuon', 'globalBeamHaloMuonEndCapslOnly', 'cosmicMuonsEndCapsOnly']
muonsBeamHaloEndCapsOnly.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsBeamHaloEndCapsOnly.fillIsolation = True
muonsBeamHaloEndCapsOnly.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5'
muonsBeamHaloEndCapsOnly.CaloExtractorPSet.CenterConeOnCalIntersection = True

# Sequences
muonrecoBeamHaloEndCapsOnly = cms.Sequence(globalBeamHaloMuonEndCapslOnly*muonsBeamHaloEndCapsOnly)

### Final sequence for endcaps only ###
muonRecoEndCapsGR = cms.Sequence(muonrecoforcosmicsEndCapsOnly*muonrecoBeamHaloEndCapsOnly)

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
muonsNoRPC = muons.clone()
muonsNoRPC.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuonsNoRPC', 'cosmicMuonsNoRPC']
muonsNoRPC.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsNoRPC.fillIsolation = True
muonsNoRPC.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5'
muonsNoRPC.CaloExtractorPSet.CenterConeOnCalIntersection = True

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

splitMuons = muons.clone()
splitMuons.inputCollectionLabels = ['splittedTracksP5', 'globalCosmicSplitMuons', 'cosmicMuons']
splitMuons.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
splitMuons.fillIsolation = True
splitMuons.TrackExtractorPSet.inputTrackCollection = 'splittedTracksP5'
splitMuons.CaloExtractorPSet.CenterConeOnCalIntersection = True

#Sequences

# Final sequence
muonrecoforsplitcosmics = cms.Sequence(globalCosmicSplitMuons*splitMuons)

##############################################

######################## LHC like Reco #############################

# Standard reco
from RecoMuon.Configuration.RecoMuon_cff import *

## Barrel only ##

# Seed generator 
lhcMuonSeedBarrelOnly = MuonSeed.clone()
lhcMuonSeedBarrelOnly.EnableCSCMeasurement = False

# Stand alone muon track producer
lhcStandAloneMuonsBarrelOnly = standAloneMuons.clone()
lhcStandAloneMuonsBarrelOnly.STATrajBuilderParameters.BWFilterParameters.EnableCSCMeasurement = False
lhcStandAloneMuonsBarrelOnly.InputObjects = 'lhcMuonSeedBarrelOnly'
#lhcStandAloneMuonsBarrelOnly.STATrajBuilderParameters.NavigationType = 'Direct'

# Muon Id producer
lhcSTAMuonsBarrelOnly = muons.clone()
lhcSTAMuonsBarrelOnly.inputCollectionLabels = ['lhcStandAloneMuonsBarrelOnly']
lhcSTAMuonsBarrelOnly.inputCollectionTypes = ['outer tracks']
lhcSTAMuonsBarrelOnly.fillIsolation = True
lhcSTAMuonsBarrelOnly.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5LHCNavigation'
lhcSTAMuonsBarrelOnly.CaloExtractorPSet.CenterConeOnCalIntersection = True

# Seqeunces
lhcMuonBarrelOnly = cms.Sequence(lhcMuonSeedBarrelOnly*lhcStandAloneMuonsBarrelOnly)

# Final sequence
muonrecocosmicLHCBarrelOnly = cms.Sequence(lhcMuonBarrelOnly*lhcSTAMuonsBarrelOnly)

##############################################

## Endcaps only ##

# Seed generator
lhcMuonSeedEndCapsOnly = MuonSeed.clone()
lhcMuonSeedEndCapsOnly.EnableDTMeasurement = False

# Stand alone muon track producer
lhcStandAloneMuonsEndCapsOnly = standAloneMuons.clone()
lhcStandAloneMuonsEndCapsOnly.STATrajBuilderParameters.BWFilterParameters.EnableDTMeasurement = False
lhcStandAloneMuonsEndCapsOnly.InputObjects = 'lhcMuonSeedEndCapsOnly'
#lhcStandAloneMuonsBarrelOnly.STATrajBuilderParameters.NavigationType = 'Direct'

# Muon Id producer
lhcSTAMuonsEndCapsOnly = muons.clone()
lhcSTAMuonsEndCapsOnly.inputCollectionLabels = ['lhcStandAloneMuonsEndCapsOnly']
lhcSTAMuonsEndCapsOnly.inputCollectionTypes = ['outer tracks']
lhcSTAMuonsEndCapsOnly.fillIsolation = True
lhcSTAMuonsEndCapsOnly.TrackExtractorPSet.inputTrackCollection = 'ctfWithMaterialTracksP5LHCNavigation'
lhcSTAMuonsEndCapsOnly.CaloExtractorPSet.CenterConeOnCalIntersection = True

# Seqeunces
lhcMuonEndCapsOnly = cms.Sequence(lhcMuonSeedEndCapsOnly*lhcStandAloneMuonsEndCapsOnly)

# Final sequence
muonrecocosmicLHCEndCapsOnly = cms.Sequence(lhcMuonEndCapsOnly*lhcSTAMuonsEndCapsOnly)

## Fianl sequence for cosmics a la LHC 
muonRecoLHC = cms.Sequence(muonrecocosmicLHCBarrelOnly*muonrecocosmicLHCEndCapsOnly)

##############################################


########################### SEQUENCE TO BE ADDED in ReconstructionGR_cff ##############################################

muonRecoGR = cms.Sequence(muonRecoAllGR*muonRecoBarrelGR*muonRecoEndCapsGR*muonrecoforcosmicsNoRPC*muonrecoforsplitcosmics*muonRecoLHC)

#######################################################################################################################





