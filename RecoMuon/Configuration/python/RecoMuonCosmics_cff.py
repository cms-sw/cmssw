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
import copy
from RecoMuon.MuonIdentification.muons_cfi import *
STAMuons = copy.deepcopy(muons)
import copy
from RecoMuon.MuonIdentification.muons_cfi import *
TKMuons = copy.deepcopy(muons)
import copy
from RecoMuon.MuonIdentification.muons_cfi import *
GLBMuons = copy.deepcopy(muons)
STAmuontrackingforcosmics = cms.Sequence(CosmicMuonSeed*cosmicMuons)
muontrackingforcosmics = cms.Sequence(STAmuontrackingforcosmics*globalCosmicMuons)
allmuons = cms.Sequence(muons*STAMuons*TKMuons*GLBMuons)
muonrecoforcosmics = cms.Sequence(muontrackingforcosmics*allmuons)
STAmuonrecoforcosmics = cms.Sequence(STAmuontrackingforcosmics*STAMuons)
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

