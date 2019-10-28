import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.gsfElectronSequence_cff import *
from RecoEgamma.EgammaElectronProducers.uncleanedOnlyElectronSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.photonSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.conversionSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.allConversionSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.gedPhotonSequence_cff import *
from RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff import *
from RecoEgamma.EgammaIsolationAlgos.interestingEgammaIsoDetIdsSequence_cff import *
from RecoEgamma.PhotonIdentification.photonId_cff import *
from RecoEgamma.ElectronIdentification.electronIdSequence_cff import *
from RecoEgamma.EgammaHFProducers.hfEMClusteringSequence_cff import *
from TrackingTools.Configuration.TrackingTools_cff import *

from RecoEgamma.EgammaIsolationAlgos.egmIsolationDefinitions_cff import *

#importing new gedGsfElectronSequence :
#from RecoEgamma.EgammaElectronProducers.gedGsfElectronSequence_cff import *
from RecoEgamma.EgammaElectronProducers.pfBasedElectronIso_cff import *

egammaGlobalRecoTask = cms.Task(electronGsfTrackingTask,conversionTrackTask,allConversionTask)
egammaGlobalReco = cms.Sequence(egammaGlobalRecoTask)
# this might be historical: not sure why we do this
from Configuration.Eras.Modifier_fastSim_cff import fastSim
_fastSim_egammaGlobalRecoTask = egammaGlobalRecoTask.copy()
_fastSim_egammaGlobalRecoTask.replace(conversionTrackTask,conversionTrackTaskNoEcalSeeded)
fastSim.toReplaceWith(egammaGlobalRecoTask, _fastSim_egammaGlobalRecoTask)

egammarecoTask = cms.Task(gsfElectronTask,conversionTask,photonTask)
egammareco = cms.Sequence(egammarecoTask)
egammaHighLevelRecoPrePFTask = cms.Task(gsfEcalDrivenElectronTask,uncleanedOnlyElectronTask,conversionTask,photonTask)
egammaHighLevelRecoPrePF = cms.Sequence(egammaHighLevelRecoPrePFTask)
# not commisoned and not relevant in FastSim (?):
fastSim.toReplaceWith(egammarecoTask, egammarecoTask.copyAndExclude([conversionTask]))
fastSim.toReplaceWith(egammaHighLevelRecoPrePFTask,egammaHighLevelRecoPrePFTask.copyAndExclude([uncleanedOnlyElectronTask,conversionTask]))

#egammaHighLevelRecoPostPF = cms.Sequence(gsfElectronMergingSequence*interestingEgammaIsoDetIds*photonIDSequence*eIdSequence*hfEMClusteringSequence)
#adding new gedGsfElectronSequence and gedPhotonSequence :
#egammaHighLevelRecoPostPF = cms.Sequence(gsfElectronMergingSequence*gedGsfElectronSequence*interestingEgammaIsoDetIds*gedPhotonSequence*photonIDSequence*eIdSequence*hfEMClusteringSequence)
egammaHighLevelRecoPostPFTask = cms.Task(interestingEgammaIsoDetIds,egmIsolationTask,photonIDTask,photonIDTaskGED,eIdTask,hfEMClusteringTask)
egammaHighLevelRecoPostPF = cms.Sequence(egammaHighLevelRecoPostPFTask)

egammarecoFullTask = cms.Task(egammarecoTask,interestingEgammaIsoDetIds,egmIsolationTask,photonIDTask,eIdTask,hfEMClusteringTask)
egammarecoFull = cms.Sequence(egammarecoFullTask)
egammarecoWithIDTask = cms.Task(egammarecoTask,photonIDTask,eIdTask)
egammarecoWithID = cms.Sequence(egammarecoWithIDTask)
egammareco_woConvPhotonsTask = cms.Task(gsfElectronTask,photonTask)
egammareco_woConvPhotons = cms.Sequence(egammareco_woConvPhotonsTask)
egammareco_withIsolationTask = cms.Task(egammarecoTask,egammaIsolationTask)
egammareco_withIsolation = cms.Sequence(egammareco_withIsolationTask)
egammareco_withIsolation_woConvPhotonsTask = cms.Task(egammareco_woConvPhotonsTask,egammaIsolationTask)
egammareco_withIsolation_woConvPhotons = cms.Sequence(egammareco_withIsolation_woConvPhotonsTask)
egammareco_withPhotonIDTask = cms.Task(egammarecoTask,photonIDTask)
egammareco_withPhotonID = cms.Sequence(egammareco_withPhotonIDTask)
egammareco_withElectronIDTask = cms.Task(egammarecoTask,eIdTask)
egammareco_withElectronID = cms.Sequence(egammareco_withElectronIDTask)

egammarecoFull_woHFElectronsTask = cms.Task(egammarecoTask,interestingEgammaIsoDetIds,photonIDTask,eIdTask)
egammarecoFull_woHFElectrons = cms.Sequence(egammarecoFull_woHFElectronsTask)

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
#HI-specific algorithms needed in pp scenario special configurations 
from RecoHI.HiEgammaAlgos.photonIsolationHIProducer_cfi import photonIsolationHIProducerpp
from RecoHI.HiEgammaAlgos.photonIsolationHIProducer_cfi import photonIsolationHIProducerppGED
from RecoHI.HiEgammaAlgos.photonIsolationHIProducer_cfi import photonIsolationHIProducerppIsland

_egammaHighLevelRecoPostPF_HITask = egammaHighLevelRecoPostPFTask.copy()
_egammaHighLevelRecoPostPF_HITask.add(photonIsolationHIProducerpp)
_egammaHighLevelRecoPostPF_HITask.add(photonIsolationHIProducerppGED)
_egammaHighLevelRecoPostPF_HITask.add(photonIsolationHIProducerppIsland)
for e in [pA_2016, peripheralPbPb, pp_on_AA_2018, pp_on_XeXe_2017, ppRef_2017]:
    e.toReplaceWith(egammaHighLevelRecoPostPFTask, _egammaHighLevelRecoPostPF_HITask)
