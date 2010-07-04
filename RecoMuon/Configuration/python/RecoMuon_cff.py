import FWCore.ParameterSet.Config as cms

# Standard pp setup
from RecoMuon.Configuration.RecoMuonPPonly_cff import *

########################################################

# Sequence for cosmic reconstruction

# Seed generator
from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *

# Stand alone muon track producer
from RecoMuon.CosmicMuonProducer.cosmicMuons_cff import *

# Global muon track producer
from RecoMuon.CosmicMuonProducer.globalCosmicMuons_cff import *

# Muon Id producer
muonsFromCosmics = RecoMuon.MuonIdentification.muons_cfi.muons.clone()

muonsFromCosmics.inputCollectionLabels = ['globalCosmicMuons', 'cosmicMuons']
muonsFromCosmics.inputCollectionTypes = ['links', 'outer tracks']
muonsFromCosmics.fillIsolation = False
muonsFromCosmics.fillGlobalTrackQuality = False

muoncosmicreco2legs = cms.Sequence(cosmicMuons*globalCosmicMuons*muonsFromCosmics)


# 1 Leg type

# Stand alone muon track producer
cosmicMuons1Leg = cosmicMuons.clone()
cosmicMuons1Leg.TrajectoryBuilderParameters.BuildTraversingMuon = True
cosmicMuons1Leg.TrajectoryBuilderParameters.Strict1Leg = True
cosmicMuons1Leg.MuonSeedCollectionLabel = 'CosmicMuonSeed'

# Global muon track producer
globalCosmicMuons1Leg = globalCosmicMuons.clone()
globalCosmicMuons1Leg.MuonCollectionLabel = 'cosmicMuons1Leg'

# Muon Id producer
muonsFromCosmics1Leg = muons.clone()
muonsFromCosmics1Leg.inputCollectionLabels = ['globalCosmicMuons1Leg', 'cosmicMuons1Leg']
muonsFromCosmics1Leg.inputCollectionTypes = ['links', 'outer tracks']
muonsFromCosmics1Leg.fillIsolation = False
muonsFromCosmics1Leg.fillGlobalTrackQuality = False

muoncosmicreco1leg = cms.Sequence(cosmicMuons1Leg*globalCosmicMuons1Leg*muonsFromCosmics1Leg)

muoncosmicreco = cms.Sequence(CosmicMuonSeed*(muoncosmicreco2legs+muoncosmicreco1leg)*cosmicsMuonIdSequence)
