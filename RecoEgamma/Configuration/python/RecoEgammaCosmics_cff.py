import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.cosmicPhotonSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.cosmicConversionSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.cosmicConversionTrackSequence_cff import *
from RecoEgamma.PhotonIdentification.photonId_cff import *

egammarecoGlobal_cosmicsTask = cms.Task(cosmicConversionTrackTask)
egammarecoGlobal_cosmics = cms.Sequence(egammarecoGlobal_cosmicsTask)
egammarecoCosmics_woElectronsTask = cms.Task(cosmicConversionTask,cosmicPhotonTask,photonIDTask)
egammarecoCosmics_woElectrons = cms.Sequence(egammarecoCosmics_woElectronsTask)
