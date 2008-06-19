import FWCore.ParameterSet.Config as cms

#
# tracker
#
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
#
# calorimeters
#
from RecoLocalCalo.Configuration.RecoLocalCalo_Cosmics_cff import *
from RecoEcal.Configuration.RecoEcalCosmics_cff import *
#
# muons
#
from RecoLocalMuon.Configuration.RecoLocalMuonCosmics_cff import *
from RecoMuon.Configuration.RecoMuonCosmics_cff import *
#
# jets and met
#
from RecoJets.Configuration.RecoCaloTowersGR_cff import *
from RecoJets.Configuration.RecoJetsGR_cff import *
from RecoMET.Configuration.RecoMET_cff import *

# local reco
trackerCosmics = cms.Sequence(offlineBeamSpot*trackerlocalreco*tracksP5)
caloCosmics = cms.Sequence(calolocalreco*ecalClusters)
muonsLocalRecoCosmics = cms.Sequence(muonlocalreco+muonlocalrecoNoDrift)

localReconstructionCosmics = cms.Sequence(trackerCosmics*caloCosmics*muonsLocalRecoCosmics)

# global reco
muonsCosmics = cms.Sequence(muonRecoGR)
jetsCosmics = cms.Sequence(recoCaloTowersGR*recoJetsGR)
metrecoCosmics = cms.Sequence(metreco)

reconstructionCosmics = cms.Sequence(localReconstructionCosmics*muonsCosmics*jetsCosmics*metrecoCosmics)

