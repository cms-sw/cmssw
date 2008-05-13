import FWCore.ParameterSet.Config as cms

# All the services like the magnitc field, the geometries and so on are included
# in the *.cff
from RecoMuon.StandAloneMuonProducer.standAloneCosmicMuons_cff import *
# Seed generator
from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *
# Stand alone muon track producer
from RecoMuon.CosmicMuonProducer.cosmicMuons_cff import *
# Global muon track producer
from RecoMuon.CosmicMuonProducer.globalCosmicMuons_cff import *
# Muon Id producer
from RecoMuon.MuonIdentification.muonIdProducerSequence_cff import *
import RecoMuon.MuonIdentification.muons_cfi
STAMuons = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonIdentification.muons_cfi
TKMuons = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonIdentification.muons_cfi
GLBMuons = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi
# Muon Isolation sequence
#include "RecoMuon/MuonIsolationProducers/data/muIsolation.cff"
#
# Muon Reconstruction plus Isolation
#sequence muonrecoforcosmics_plus_isolation = {muonrecoforcosmics,muIsolation}
# Alternative cosmic muon reconstruction
# Temp sequences until the endcaps are not close to the barrel  ###
# Only barrel
CosmicMuonSeedBarrelOnly = RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi.CosmicMuonSeed.clone()
import RecoMuon.CosmicMuonProducer.cosmicMuons_cfi
cosmicMuonsBarrelOnly = RecoMuon.CosmicMuonProducer.cosmicMuons_cfi.cosmicMuons.clone()
import RecoMuon.CosmicMuonProducer.globalCosmicMuons_cfi
globalCosmicMuonsBarrelOnly = RecoMuon.CosmicMuonProducer.globalCosmicMuons_cfi.globalCosmicMuons.clone()
import RecoMuon.MuonIdentification.muons_cfi
STAMuonsBarrelOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonIdentification.muons_cfi
GLBMuonsBarrelOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonIdentification.muons_cfi
muonsBarrelOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi
# Only endcaps
CosmicMuonSeedEndCapsOnly = RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi.CosmicMuonSeed.clone()
import RecoMuon.CosmicMuonProducer.cosmicMuons_cfi
cosmicMuonsEndCapsOnly = RecoMuon.CosmicMuonProducer.cosmicMuons_cfi.cosmicMuons.clone()
import RecoMuon.MuonIdentification.muons_cfi
STAMuonsEndCapsOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
STAmuontrackingforcosmics = cms.Sequence(CosmicMuonSeed*cosmicMuons)
muontrackingforcosmics = cms.Sequence(STAmuontrackingforcosmics*globalCosmicMuons)
allmuons = cms.Sequence(muons*STAMuons*TKMuons*GLBMuons)
muonrecoforcosmics = cms.Sequence(muontrackingforcosmics*allmuons)
STAmuonrecoforcosmics = cms.Sequence(STAmuontrackingforcosmics*STAMuons)
STAmuontrackingforcosmicsBarrelOnly = cms.Sequence(CosmicMuonSeedBarrelOnly*cosmicMuonsBarrelOnly)
muontrackingforcosmicsBarrelOnly = cms.Sequence(STAmuontrackingforcosmicsBarrelOnly*globalCosmicMuonsBarrelOnly)
allmuonsBarrelOnly = cms.Sequence(muonsBarrelOnly*STAMuonsBarrelOnly*GLBMuonsBarrelOnly)
muonrecoforcosmicsBarrelOnly = cms.Sequence(muontrackingforcosmicsBarrelOnly*allmuonsBarrelOnly)
STAmuonrecoforcosmicsBarrelOnly = cms.Sequence(STAmuontrackingforcosmicsBarrelOnly*STAMuonsBarrelOnly)
STAmuontrackingforcosmicsEnsCapsOnly = cms.Sequence(CosmicMuonSeedEndCapsOnly*cosmicMuonsEndCapsOnly)
muontrackingforcosmicsEndCapsOnly = cms.Sequence(STAmuontrackingforcosmicsEnsCapsOnly)
allmuonsEndCapsOnly = cms.Sequence(STAMuonsEndCapsOnly)
muonrecoforcosmicsEndCapsOnly = cms.Sequence(muontrackingforcosmicsEndCapsOnly*allmuonsEndCapsOnly)
STAmuonrecoforcosmicsEndCapsOnly = cms.Sequence(STAmuontrackingforcosmicsEnsCapsOnly*STAMuonsEndCapsOnly)
globalCosmicMuons.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'
muons.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuons', 'cosmicMuons']
muons.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muons.fillIsolation = False
STAMuons.inputCollectionLabels = ['cosmicMuons']
STAMuons.inputCollectionTypes = ['outer tracks']
STAMuons.fillIsolation = False
TKMuons.inputCollectionLabels = ['ctfWithMaterialTracksP5']
TKMuons.inputCollectionTypes = ['inner tracks']
TKMuons.fillIsolation = False
GLBMuons.inputCollectionLabels = ['globalCosmicMuons']
GLBMuons.inputCollectionTypes = ['links']
GLBMuons.fillIsolation = False
CosmicMuonSeedBarrelOnly.EnableCSCMeasurement = False
cosmicMuonsBarrelOnly.TrajectoryBuilderParameters.EnableCSCMeasurement = False
globalCosmicMuonsBarrelOnly.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'
globalCosmicMuonsBarrelOnly.MuonCollectionLabel = 'cosmicMuonsBarrelOnly'
STAMuonsBarrelOnly.inputCollectionLabels = ['cosmicMuonsBarrelOnly']
STAMuonsBarrelOnly.inputCollectionTypes = ['outer tracks']
STAMuonsBarrelOnly.fillIsolation = False
GLBMuonsBarrelOnly.inputCollectionLabels = ['globalCosmicMuonsBarrelOnly']
GLBMuonsBarrelOnly.inputCollectionTypes = ['links']
GLBMuonsBarrelOnly.fillIsolation = False
muonsBarrelOnly.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuonsBarrelOnly', 'cosmicMuonsBarrelOnly']
muonsBarrelOnly.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsBarrelOnly.fillIsolation = False
CosmicMuonSeedEndCapsOnly.EnableDTMeasurement = False
cosmicMuonsEndCapsOnly.TrajectoryBuilderParameters.EnableDTMeasurement = False
STAMuonsEndCapsOnly.inputCollectionLabels = ['cosmicMuonsEndCapsOnly']
STAMuonsEndCapsOnly.inputCollectionTypes = ['outer tracks']
STAMuonsEndCapsOnly.fillIsolation = False

