import FWCore.ParameterSet.Config as cms
#
# luminosity
#
from RecoLuminosity.LumiProducer.lumiProducer_cff import *
#
# tracker
#
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.Configuration.RecoTrackerBHM_cff import *

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
from RecoVertex.Configuration.RecoVertexCosmicTracks_cff import *

#
# jets and met
#
from RecoJets.Configuration.RecoCaloTowersGR_cff import *
from RecoJets.Configuration.RecoJetsGR_cff import *
from RecoMET.Configuration.RecoMET_Cosmics_cff import *

#
## egamma
#
from RecoEgamma.Configuration.RecoEgammaCosmics_cff import *

# local reco
trackerCosmics = cms.Sequence(offlineBeamSpot*trackerlocalreco*tracksP5)
hbhereco = hbheprereco.clone()
calolocalreco.replace(hbheprereco,hbhereco)
caloCosmics = cms.Sequence(calolocalreco*ecalClusters)
caloCosmics_HcalNZS = cms.Sequence(calolocalrecoNZS*ecalClusters)
muonsLocalRecoCosmics = cms.Sequence(muonlocalreco+muonlocalrecoT0Seg)

localReconstructionCosmics         = cms.Sequence(trackerCosmics*caloCosmics*muonsLocalRecoCosmics*vertexrecoCosmics+lumiProducer)
localReconstructionCosmics_HcalNZS = cms.Sequence(trackerCosmics*caloCosmics_HcalNZS*muonsLocalRecoCosmics*vertexrecoCosmics +lumiProducer)


# global reco
muonsCosmics = cms.Sequence(muonRecoGR)
jetsCosmics = cms.Sequence(recoCaloTowersGR*recoJetsGR)
egammaCosmics = cms.Sequence(egammarecoGlobal_cosmics*egammarecoCosmics_woElectrons)


from FWCore.Modules.logErrorHarvester_cfi import *


reconstructionCosmics         = cms.Sequence(localReconstructionCosmics*
                                             beamhaloTracksSeq*
                                             jetsCosmics*
                                             muonsCosmics*
                                             regionalTrackerCosmic*
                                             metrecoCosmics*
                                             egammaCosmics*
                                             logErrorHarvester)
reconstructionCosmics_HcalNZS = cms.Sequence(localReconstructionCosmics_HcalNZS*
                                             beamhaloTracksSeq*
                                             jetsCosmics*
                                             muonsCosmics*
                                             regionalTrackerCosmic*
                                             metrecoCosmics*
                                             egammaCosmics*
                                             logErrorHarvester)
reconstructionCosmics_woTkBHM = cms.Sequence(localReconstructionCosmics*
                                             jetsCosmics*
                                             muonsCosmics*
                                             regionalTrackerCosmic*
                                             metrecoCosmics*
                                             egammaCosmics)
