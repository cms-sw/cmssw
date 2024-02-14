import FWCore.ParameterSet.Config as cms
#
# luminosity
#
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
from RecoHGCal.Configuration.recoHGCAL_cff import *

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

# ugly hack: for the time being remove tracking (no Cosmics seeding in Phase-2) 
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(trackerCosmicsTask,trackerCosmicsTask.copyAndExclude([tracksP5Task]))

trackerCosmics = cms.Sequence(trackerCosmicsTask)
caloCosmicsTask = cms.Task(calolocalrecoTaskCosmics,ecalClustersCosmicsTask)
caloCosmics = cms.Sequence(caloCosmicsTask)
caloCosmics_HcalNZSTask = cms.Task(calolocalrecoTaskCosmicsNZS,ecalClustersCosmicsTask)
caloCosmics_HcalNZS = cms.Sequence(caloCosmics_HcalNZSTask)
muonsLocalRecoCosmicsTask = cms.Task(muonlocalrecoTask,muonlocalrecoT0SegTask)
muonsLocalRecoCosmics = cms.Sequence(muonsLocalRecoCosmicsTask)
localReconstructionCosmicsTask = cms.Task(bunchSpacingProducer,trackerCosmicsTask,caloCosmicsTask,muonsLocalRecoCosmicsTask,vertexrecoCosmicsTask)
#phase2_tracker.toReplaceWith(localReconstructionCosmicsTask,localReconstructionCosmicsTask.copyAndExclude([vertexrecoCosmicsTask]))  

localReconstructionCosmics         = cms.Sequence(localReconstructionCosmicsTask)
localReconstructionCosmics_HcalNZSTask = cms.Task(bunchSpacingProducer,trackerCosmicsTask,caloCosmics_HcalNZSTask,muonsLocalRecoCosmicsTask,vertexrecoCosmicsTask)
localReconstructionCosmics_HcalNZS = cms.Sequence(localReconstructionCosmics_HcalNZSTask)

# global reco
muonsCosmicsTask = cms.Task(muonRecoGRTask)
jetsCosmicsTask = cms.Task(recoCaloTowersGRTask,recoJetsGRTask)
egammaCosmicsTask = cms.Task(egammarecoGlobal_cosmicsTask,egammarecoCosmics_woElectronsTask)

from FWCore.Modules.logErrorHarvester_cfi import *

reconstructionCosmicsTask = cms.Task(localReconstructionCosmicsTask,
                                     beamhaloTracksTask,
                                     jetsCosmicsTask,
                                     muonsCosmicsTask,
                                     regionalCosmicTracksTask,
                                     cosmicDCTracksSeqTask,
                                     metrecoCosmicsTask,
                                     egammaCosmicsTask,
                                     logErrorHarvester)

# ugly hack
# for the time being remove all tasks related to tracking
phase2_tracker.toReplaceWith(reconstructionCosmicsTask,reconstructionCosmicsTask.copyAndExclude([beamhaloTracksTask,
                                                                                                 cosmicDCTracksSeqTask,
                                                                                                 regionalCosmicTracksTask,
                                                                                                 metrecoCosmicsTask,
                                                                                                 muonsCosmicsTask]))

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
_phase2HGALRecoTask = reconstructionCosmicsTask.copy()
_phase2HGALRecoTask.add(iterTICLTask)
phase2_hgcal.toReplaceWith(reconstructionCosmicsTask, _phase2HGALRecoTask)

from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
_phase2HFNoseRecoTask = reconstructionCosmicsTask.copy()
_phase2HFNoseRecoTask.add(iterHFNoseTICLTask)
phase2_hfnose.toReplaceWith(reconstructionCosmicsTask, _phase2HFNoseRecoTask)

reconstructionCosmics = cms.Sequence(reconstructionCosmicsTask)
#logErrorHarvester should only wait for items produced in the reconstructionCosmics sequence
_modulesInReconstruction = list()
reconstructionCosmics.visit(cms.ModuleNamesFromGlobalsVisitor(globals(),_modulesInReconstruction))
logErrorHarvester.includeModules = cms.untracked.vstring(sorted(set(_modulesInReconstruction)))

reconstructionCosmics_HcalNZSTask = cms.Task(localReconstructionCosmics_HcalNZSTask,
                                             beamhaloTracksTask,
                                             jetsCosmicsTask,
                                             muonsCosmicsTask,
                                             regionalCosmicTracksTask,
                                             cosmicDCTracksSeqTask,
                                             metrecoCosmicsTask,
                                             egammaCosmicsTask,
                                             logErrorHarvester)

phase2_tracker.toReplaceWith(reconstructionCosmics_HcalNZSTask,reconstructionCosmics_HcalNZSTask.copyAndExclude([beamhaloTracksTask,cosmicDCTracksSeqTask,regionalCosmicTracksTask]))
reconstructionCosmics_HcalNZS = cms.Sequence(reconstructionCosmics_HcalNZSTask)

reconstructionCosmics_woTkBHMTask = cms.Task(localReconstructionCosmicsTask,
                                             jetsCosmicsTask,
                                             muonsCosmicsTask,
                                             regionalCosmicTracksTask,
                                             cosmicDCTracksSeqTask,
                                             metrecoCosmicsTask,
                                             egammaCosmicsTask)

phase2_tracker.toReplaceWith(reconstructionCosmics_woTkBHMTask,reconstructionCosmics_woTkBHMTask.copyAndExclude([beamhaloTracksTask,cosmicDCTracksSeqTask,regionalCosmicTracksTask]))  
reconstructionCosmics_woTkBHM = cms.Sequence(reconstructionCosmics_woTkBHMTask)
