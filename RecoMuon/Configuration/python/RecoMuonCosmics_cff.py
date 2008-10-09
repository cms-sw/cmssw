import FWCore.ParameterSet.Config as cms

######################## Cosmic Reco #############################

## Full detector ##

# Seed generator
from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *

# Stand alone muon track producer
from RecoMuon.CosmicMuonProducer.cosmicMuons_cff import *

# Global muon track producer
from RecoMuon.CosmicMuonProducer.globalCosmicMuons_cff import *
globalCosmicMuons.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'

# Muon Id producer
from RecoMuon.MuonIdentification.muonIdProducerSequence_cff import *

muons.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuons', 'cosmicMuons']
muons.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muons.fillIsolation = False

STAMuons = muons.clone()
STAMuons.inputCollectionLabels = ['cosmicMuons']
STAMuons.inputCollectionTypes = ['outer tracks']
STAMuons.fillIsolation = False

TKMuons = muons.clone()
TKMuons.inputCollectionLabels = ['ctfWithMaterialTracksP5']
TKMuons.inputCollectionTypes = ['inner tracks']
TKMuons.fillIsolation = False

GLBMuons = muons.clone()
GLBMuons.inputCollectionLabels = ['globalCosmicMuons']
GLBMuons.inputCollectionTypes = ['links']
GLBMuons.fillIsolation = False


## Sequences

# Stand Alone Tracking
STAmuontrackingforcosmics = cms.Sequence(CosmicMuonSeed*cosmicMuons)
# Stand Alone Tracking plus muon ID
STAmuonrecoforcosmics = cms.Sequence(STAmuontrackingforcosmics*STAMuons)

# Stand Alone Tracking plus global tracking
muontrackingforcosmics = cms.Sequence(STAmuontrackingforcosmics*globalCosmicMuons)

# all muons id
allmuons = cms.Sequence(muons*STAMuons*TKMuons*GLBMuons)

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
STAMuonsBarrelOnly = muons.clone()
STAMuonsBarrelOnly.inputCollectionLabels = ['cosmicMuonsBarrelOnly']
STAMuonsBarrelOnly.inputCollectionTypes = ['outer tracks']
STAMuonsBarrelOnly.fillIsolation = False

GLBMuonsBarrelOnly = muons.clone()
GLBMuonsBarrelOnly.inputCollectionLabels = ['globalCosmicMuonsBarrelOnly']
GLBMuonsBarrelOnly.inputCollectionTypes = ['links']
GLBMuonsBarrelOnly.fillIsolation = False

muonsBarrelOnly = muons.clone()
muonsBarrelOnly.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuonsBarrelOnly', 'cosmicMuonsBarrelOnly']
muonsBarrelOnly.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsBarrelOnly.fillIsolation = False

#Sequences

# Stand Alone Tracking
STAmuontrackingforcosmicsBarrelOnly = cms.Sequence(CosmicMuonSeedBarrelOnly*cosmicMuonsBarrelOnly)

# Stand Alone Tracking plus global tracking
muontrackingforcosmicsBarrelOnly = cms.Sequence(STAmuontrackingforcosmicsBarrelOnly*globalCosmicMuonsBarrelOnly)

# Stand Alone Tracking plus muon ID
STAmuonrecoforcosmicsBarrelOnly = cms.Sequence(STAmuontrackingforcosmicsBarrelOnly*STAMuonsBarrelOnly)

# all muons id
allmuonsBarrelOnly = cms.Sequence(muonsBarrelOnly*STAMuonsBarrelOnly*GLBMuonsBarrelOnly)

# Final sequence
muonrecoforcosmicsBarrelOnly = cms.Sequence(muontrackingforcosmicsBarrelOnly*allmuonsBarrelOnly)

########

# 1 leg mode

# Stand alone muon track producer
cosmicMuons1Leg = cosmicMuons.clone()
#cosmicMuons1Leg.TrajectoryBuilderParameters.EnableCSCMeasurement = False
#cosmicMuons1Leg.TrajectoryBuilderParameters.MuonNavigationParameters.Endcap = False
cosmicMuons1Leg.TrajectoryBuilderParameters.BuildTraversingMuon = True
cosmicMuons1Leg.MuonSeedCollectionLabel = 'CosmicMuonSeedBarrelOnly'

# Global muon track producer
globalCosmicMuons1Leg = globalCosmicMuons.clone()
globalCosmicMuons1Leg.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'
globalCosmicMuons1Leg.MuonCollectionLabel = 'cosmicMuons1Leg'

# Muon Id producer
STAMuons1Leg = muons.clone()
STAMuons1Leg.inputCollectionLabels = ['cosmicMuons1Leg']
STAMuons1Leg.inputCollectionTypes = ['outer tracks']
STAMuons1Leg.fillIsolation = False

GLBMuons1Leg = muons.clone()
GLBMuons1Leg.inputCollectionLabels = ['globalCosmicMuons1Leg']
GLBMuons1Leg.inputCollectionTypes = ['links']
GLBMuons1Leg.fillIsolation = False

muons1Leg = muons.clone()
muons1Leg.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuons1Leg', 'cosmicMuons1Leg']
muons1Leg.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muons1Leg.fillIsolation = False

# Sequences

# Stand Alone Tracking
STAmuontrackingforcosmics1Leg = cms.Sequence(CosmicMuonSeedBarrelOnly*cosmicMuons1Leg)

# Stand Alone Tracking plus global tracking
muontrackingforcosmics1Leg = cms.Sequence(STAmuontrackingforcosmics1Leg*globalCosmicMuons1Leg)

# all muons id
allmuons1Leg = cms.Sequence(muons1Leg*STAMuons1Leg*GLBMuons1Leg)

# Stand Alone Tracking plus muon ID
STAmuonrecoforcosmics1Leg = cms.Sequence(STAmuontrackingforcosmics1Leg*STAMuons1Leg)

# Final sequence
muonrecoforcosmics1Leg = cms.Sequence(muontrackingforcosmics1Leg*allmuons1Leg)

########

# Barrel only, No drift local reco

# Seed generator
CosmicMuonSeedNoDriftBarrelOnly = CosmicMuonSeed.clone()
CosmicMuonSeedNoDriftBarrelOnly.EnableCSCMeasurement = False
CosmicMuonSeedNoDriftBarrelOnly.DTRecSegmentLabel = 'dt4DSegmentsNoDrift'

# Stand alone muon track producer
cosmicMuonsNoDriftBarrelOnly = cosmicMuons.clone()
cosmicMuonsNoDriftBarrelOnly.TrajectoryBuilderParameters.EnableCSCMeasurement = False
cosmicMuonsNoDriftBarrelOnly.TrajectoryBuilderParameters.MuonNavigationParameters.Endcap = False
cosmicMuonsNoDriftBarrelOnly.TrajectoryBuilderParameters.BuildTraversingMuon = True
cosmicMuonsNoDriftBarrelOnly.MuonSeedCollectionLabel = 'CosmicMuonSeedNoDriftBarrelOnly'
cosmicMuonsNoDriftBarrelOnly.TrajectoryBuilderParameters.DTRecSegmentLabel = 'dt4DSegmentsNoDrift'

# Global muon track producer
globalCosmicMuonsNoDriftBarrelOnly = globalCosmicMuons.clone()
globalCosmicMuonsNoDriftBarrelOnly.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'
globalCosmicMuonsNoDriftBarrelOnly.MuonCollectionLabel = 'cosmicMuonsNoDriftBarrelOnly'

# Muon Id producer
STAMuonsNoDriftBarrelOnly = muons.clone()
STAMuonsNoDriftBarrelOnly.inputCollectionLabels = ['cosmicMuonsNoDriftBarrelOnly']
STAMuonsNoDriftBarrelOnly.inputCollectionTypes = ['outer tracks']
STAMuonsNoDriftBarrelOnly.fillIsolation = False

GLBMuonsNoDriftBarrelOnly = muons.clone()
GLBMuonsNoDriftBarrelOnly.inputCollectionLabels = ['globalCosmicMuonsNoDriftBarrelOnly']
GLBMuonsNoDriftBarrelOnly.inputCollectionTypes = ['links']
GLBMuonsNoDriftBarrelOnly.fillIsolation = False

muonsNoDriftBarrelOnly = muons.clone()
muonsNoDriftBarrelOnly.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuonsNoDriftBarrelOnly', 'cosmicMuonsNoDriftBarrelOnly']
muonsNoDriftBarrelOnly.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsNoDriftBarrelOnly.fillIsolation = False

#Sequences

# Stand Alone Tracking
STAmuontrackingforcosmicsNoDriftBarrelOnly = cms.Sequence(CosmicMuonSeedNoDriftBarrelOnly*cosmicMuonsNoDriftBarrelOnly)

# Stand Alone Tracking plus global tracking
muontrackingforcosmicsNoDriftBarrelOnly = cms.Sequence(STAmuontrackingforcosmicsNoDriftBarrelOnly*globalCosmicMuonsNoDriftBarrelOnly)

# Stand Alone Tracking plus muon ID
STAmuonrecoforcosmicsNoDriftBarrelOnly = cms.Sequence(STAmuontrackingforcosmicsNoDriftBarrelOnly*STAMuonsNoDriftBarrelOnly)

# all muons id
allmuonsNoDriftBarrelOnly = cms.Sequence(muonsNoDriftBarrelOnly*STAMuonsNoDriftBarrelOnly*GLBMuonsNoDriftBarrelOnly)

# Final sequence
muonrecoforcosmicsNoDriftBarrelOnly = cms.Sequence(muontrackingforcosmicsNoDriftBarrelOnly*allmuonsNoDriftBarrelOnly)

### Final sequence for barrel only ###
muonRecoBarrelGR = cms.Sequence(muonrecoforcosmicsBarrelOnly+muonrecoforcosmics1Leg+muonrecoforcosmicsNoDriftBarrelOnly)

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
STAMuonsEndCapsOnly = muons.clone()
STAMuonsEndCapsOnly.inputCollectionLabels = ['cosmicMuonsEndCapsOnly']
STAMuonsEndCapsOnly.inputCollectionTypes = ['outer tracks']
STAMuonsEndCapsOnly.fillIsolation = False

GLBMuonsEndCapsOnly = muons.clone()
GLBMuonsEndCapsOnly.inputCollectionLabels = ['globalCosmicMuonsEndCapsOnly']
GLBMuonsEndCapsOnly.inputCollectionTypes = ['links']
GLBMuonsEndCapsOnly.fillIsolation = False

muonsEndCapsOnly = muons.clone()
muonsEndCapsOnly.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuonsEndCapsOnly', 'cosmicMuonsEndCapsOnly']
muonsEndCapsOnly.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsEndCapsOnly.fillIsolation = False

# Sequences

# Stand Alone Tracking
STAmuontrackingforcosmicsEnsCapsOnly = cms.Sequence(CosmicMuonSeedEndCapsOnly*cosmicMuonsEndCapsOnly)

# Stand Alone Tracking plus global tracking
muontrackingforcosmicsEndCapsOnly = cms.Sequence(STAmuontrackingforcosmicsEnsCapsOnly*globalCosmicMuonsEndCapsOnly)

# Stand Alone Tracking plus muon ID
STAmuonrecoforcosmicsEndCapsOnly = cms.Sequence(STAmuontrackingforcosmicsEnsCapsOnly*STAMuonsEndCapsOnly)

# all muons id
allmuonsEndCapsOnly = cms.Sequence(muonsEndCapsOnly*STAMuonsEndCapsOnly*GLBMuonsEndCapsOnly)

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
muonsBeamHaloEndCapsOnly.fillIsolation = False

# Sequences
muonrecoBeamHaloEndCapsOnly = cms.Sequence(globalBeamHaloMuonEndCapslOnly*muonsBeamHaloEndCapsOnly)

### Final sequence for endcaps only ###
muonRecoEndCapsGR = cms.Sequence(muonrecoforcosmicsEndCapsOnly*muonrecoBeamHaloEndCapsOnly)

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
lhcSTAMuonsBarrelOnly.fillIsolation = False

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
lhcSTAMuonsEndCapsOnly.fillIsolation = False

# Seqeunces
lhcMuonEndCapsOnly = cms.Sequence(lhcMuonSeedEndCapsOnly*lhcStandAloneMuonsEndCapsOnly)

# Final sequence
muonrecocosmicLHCEndCapsOnly = cms.Sequence(lhcMuonEndCapsOnly*lhcSTAMuonsEndCapsOnly)

## Fianl sequence for cosmics a la LHC 
muonRecoLHC = cms.Sequence(muonrecocosmicLHCBarrelOnly*muonrecocosmicLHCEndCapsOnly)

##############################################


########################### SEQUENCE TO BE ADDED in ReconstructionGR_cff ##############################################

muonRecoGR = cms.Sequence(muonRecoAllGR*muonRecoBarrelGR*muonRecoEndCapsGR*muonRecoLHC)

#######################################################################################################################





