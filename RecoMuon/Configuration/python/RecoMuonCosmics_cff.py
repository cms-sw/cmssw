import FWCore.ParameterSet.Config as cms

# All the services like the magnitc field, the geometries and so on are included
# in the *.cff
from RecoMuon.StandAloneMuonProducer.standAloneCosmicMuons_cff import *
# Seed generator
from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *
# Stand alone muon track producer
from RecoMuon.CosmicMuonProducer.cosmicMuons_cff import *
# Muon Id producer
from RecoMuon.MuonIdentification.muonIdProducerSequence_cff import *
# Global muon track producer
# include "RecoMuon/CosmicMuonProducer/data/globalCosmicMuons.cff"
muontrackingforcosmics = cms.Sequence(CosmicMuonSeed*cosmicMuons)
muonrecoforcosmics = cms.Sequence(muontrackingforcosmics*muons)
muons.inputCollectionLabels = ['cosmicMuons']
muons.inputCollectionTypes = ['outer tracks']
muons.fillIsolation = False

