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

egammaGlobalReco = cms.Sequence(electronGsfTracking*conversionTrackSequence*allConversionSequence)
# this might be historical: not sure why we do this
from Configuration.Eras.Modifier_fastSim_cff import fastSim
_fastSim_egammaGlobalReco = egammaGlobalReco.copy()
_fastSim_egammaGlobalReco.replace(conversionTrackSequence,conversionTrackSequenceNoEcalSeeded)
fastSim.toReplaceWith(egammaGlobalReco, _fastSim_egammaGlobalReco)

egammareco = cms.Sequence(gsfElectronSequence*conversionSequence*photonSequence)
egammaHighLevelRecoPrePF = cms.Sequence(gsfEcalDrivenElectronSequence*uncleanedOnlyElectronSequence*conversionSequence*photonSequence)
# not commisoned and not relevant in FastSim (?):
fastSim.toReplaceWith(egammareco, egammareco.copyAndExclude([conversionSequence]))
fastSim.toReplaceWith(egammaHighLevelRecoPrePF,egammaHighLevelRecoPrePF.copyAndExclude([uncleanedOnlyElectronSequence,conversionSequence]))

#egammaHighLevelRecoPostPF = cms.Sequence(gsfElectronMergingSequence*interestingEgammaIsoDetIds*photonIDSequence*eIdSequence*hfEMClusteringSequence)
#adding new gedGsfElectronSequence and gedPhotonSequence :
#egammaHighLevelRecoPostPF = cms.Sequence(gsfElectronMergingSequence*gedGsfElectronSequence*interestingEgammaIsoDetIds*gedPhotonSequence*photonIDSequence*eIdSequence*hfEMClusteringSequence)
egammaHighLevelRecoPostPF = cms.Sequence(interestingEgammaIsoDetIds*egmIsolationSequence*photonIDSequence*photonIDSequenceGED*eIdSequence*hfEMClusteringSequence)


egammarecoFull = cms.Sequence(egammareco*interestingEgammaIsoDetIds*egmIsolationSequence*photonIDSequence*eIdSequence*hfEMClusteringSequence)
egammarecoWithID = cms.Sequence(egammareco*photonIDSequence*eIdSequence)
egammareco_woConvPhotons = cms.Sequence(gsfElectronSequence*photonSequence)
egammareco_withIsolation = cms.Sequence(egammareco*egammaIsolationSequence)
egammareco_withIsolation_woConvPhotons = cms.Sequence(egammareco_woConvPhotons*egammaIsolationSequence)
egammareco_withPhotonID = cms.Sequence(egammareco*photonIDSequence)
egammareco_withElectronID = cms.Sequence(egammareco*eIdSequence)

egammarecoFull_woHFElectrons = cms.Sequence(egammareco*interestingEgammaIsoDetIds*photonIDSequence*eIdSequence)

from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_ppRef_2017_cff import ppRef_2017
#HI-specific algorithms needed in pp scenario special configurations 
from RecoHI.HiEgammaAlgos.photonIsolationHIProducer_cfi import photonIsolationHIProducerpp
from RecoHI.HiEgammaAlgos.photonIsolationHIProducer_cfi import photonIsolationHIProducerppGED
from RecoHI.HiEgammaAlgos.photonIsolationHIProducer_cfi import photonIsolationHIProducerppIsland

_egammaHighLevelRecoPostPF_HI = egammaHighLevelRecoPostPF.copy()
_egammaHighLevelRecoPostPF_HI += photonIsolationHIProducerpp
_egammaHighLevelRecoPostPF_HI += photonIsolationHIProducerppGED
_egammaHighLevelRecoPostPF_HI += photonIsolationHIProducerppIsland
for e in [pA_2016, peripheralPbPb, pp_on_AA_2018, pp_on_XeXe_2017, ppRef_2017]:
    e.toReplaceWith(egammaHighLevelRecoPostPF, _egammaHighLevelRecoPostPF_HI)
