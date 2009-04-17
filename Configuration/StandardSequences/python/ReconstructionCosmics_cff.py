import FWCore.ParameterSet.Config as cms

#
# tracker
#
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.Configuration.RecoTrackerBHM_cff import *
from RecoTracker.DeDx.dedxEstimators_Cosmics_cff import *


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

# primary vertex
#from RecoVertex.Configuration.RecoVertexCosmicTracks_cff import *

#
# jets and met
#
from RecoJets.Configuration.RecoCaloTowersGR_cff import *
from RecoJets.Configuration.RecoJetsGR_cff import *
from RecoMET.Configuration.RecoMET_cff import *

#
## egamma
#
from RecoEgamma.Configuration.RecoEgammaCosmics_cff import *

# local reco
trackerCosmics = cms.Sequence(offlineBeamSpot*trackerlocalreco*tracksP5)
caloCosmics = cms.Sequence(calolocalreco*ecalClusters)
muonsLocalRecoCosmics = cms.Sequence(muonlocalreco+muonlocalrecoT0Seg)

#localReconstructionCosmics = cms.Sequence(trackerCosmics*caloCosmics*muonsLocalRecoCosmics*vertexrecoCosmics)
localReconstructionCosmics = cms.Sequence(trackerCosmics*caloCosmics*muonsLocalRecoCosmics)

# global reco
muonsCosmics = cms.Sequence(muonRecoGR)
jetsCosmics = cms.Sequence(recoCaloTowersGR*recoJetsGR)
metrecoCosmics = cms.Sequence(metreco)
egammaCosmics = cms.Sequence(egammarecoCosmics_woElectrons)


reconstructionCosmics = cms.Sequence(localReconstructionCosmics*tracksBeamHaloMuon*muonsCosmics*jetsCosmics*metrecoCosmics*egammaCosmics*doAlldEdXEstimators)

reconstructionCosmics_woDeDx = cms.Sequence(localReconstructionCosmics*tracksBeamHaloMuon*muonsCosmics*jetsCosmics*metrecoCosmics*egammaCosmics)
reconstructionCosmics_woTkBHM = cms.Sequence(localReconstructionCosmics*muonsCosmics*jetsCosmics*metrecoCosmics*egammaCosmics)
