import FWCore.ParameterSet.Config as cms
#
# luminosity
#
from RecoLuminosity.LumiProducer.lumiProducer_cff import *
from RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi import *
# no bunchspacing in cosmics
bunchSpacingProducer.overrideBunchSpacing= cms.bool(True)
bunchSpacingProducer.bunchSpacingOverride= cms.uint32(50)

#
# tracker
#
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *
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
trackerCosmicsTask = cms.Task(offlineBeamSpot,trackerlocalrecoTask,MeasurementTrackerEvent,tracksP5Task)
trackerCosmics = cms.Sequence(trackerCosmicsTask)
caloCosmicsTask = cms.Task(calolocalrecoTaskCosmics,ecalClustersCosmicsTask)
caloCosmics = cms.Sequence(caloCosmicsTask)
caloCosmics_HcalNZSTask = cms.Task(calolocalrecoTaskCosmicsNZS,ecalClustersCosmicsTask)
caloCosmics_HcalNZS = cms.Sequence(caloCosmics_HcalNZSTask)
muonsLocalRecoCosmicsTask = cms.Task(muonlocalrecoTask,muonlocalrecoT0SegTask)
muonsLocalRecoCosmics = cms.Sequence(muonsLocalRecoCosmicsTask)

localReconstructionCosmicsTask         = cms.Task(bunchSpacingProducer,trackerCosmicsTask,caloCosmicsTask,muonsLocalRecoCosmicsTask,vertexrecoCosmicsTask,lumiProducer)
localReconstructionCosmics         = cms.Sequence(localReconstructionCosmicsTask)
localReconstructionCosmics_HcalNZSTask = cms.Task(bunchSpacingProducer,trackerCosmicsTask,caloCosmics_HcalNZSTask,muonsLocalRecoCosmicsTask,vertexrecoCosmicsTask,lumiProducer)
localReconstructionCosmics_HcalNZS = cms.Sequence(localReconstructionCosmics_HcalNZSTask)


# global reco
muonsCosmicsTask = cms.Task(muonRecoGRTask)
jetsCosmicsTask = cms.Task(recoCaloTowersGRTask,recoJetsGRTask)
egammaCosmicsTask = cms.Task(egammarecoGlobal_cosmicsTask,egammarecoCosmics_woElectronsTask)


from FWCore.Modules.logErrorHarvester_cfi import *


reconstructionCosmicsTask         = cms.Task(localReconstructionCosmicsTask,
                                             beamhaloTracksTask,
                                             jetsCosmicsTask,
                                             muonsCosmicsTask,
                                             regionalCosmicTracksTask,
                                             cosmicDCTracksSeqTask,
                                             metrecoCosmicsTask,
                                             egammaCosmicsTask,
                                             logErrorHarvester)
reconstructionCosmics         = cms.Sequence(reconstructionCosmicsTask)
#logErrorHarvester should only wait for items produced in the reconstructionCosmics sequence
_modulesInReconstruction = list()
reconstructionCosmics.visit(cms.ModuleNamesFromGlobalsVisitor(globals(),_modulesInReconstruction))
logErrorHarvester.includeModules = cms.untracked.vstring(set(_modulesInReconstruction))

reconstructionCosmics_HcalNZSTask = cms.Task(localReconstructionCosmics_HcalNZSTask,
                                             beamhaloTracksTask,
                                             jetsCosmicsTask,
                                             muonsCosmicsTask,
                                             regionalCosmicTracksTask,
                                             cosmicDCTracksSeqTask,
                                             metrecoCosmicsTask,
                                             egammaCosmicsTask,
                                             logErrorHarvester)
reconstructionCosmics_HcalNZS = cms.Sequence(reconstructionCosmics_HcalNZSTask)
reconstructionCosmics_woTkBHMTask = cms.Task(localReconstructionCosmicsTask,
                                             jetsCosmicsTask,
                                             muonsCosmicsTask,
                                             regionalCosmicTracksTask,
                                             cosmicDCTracksSeqTask,
                                             metrecoCosmicsTask,
                                             egammaCosmicsTask)
reconstructionCosmics_woTkBHM = cms.Sequence(reconstructionCosmics_woTkBHMTask)
