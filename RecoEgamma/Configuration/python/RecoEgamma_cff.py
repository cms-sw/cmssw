import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.electronSequence_cff import *
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

#importing new gedGsfElectronSequence :
from RecoEgamma.EgammaElectronProducers.gedGsfElectronSequence_cff import *

egammaGlobalReco = cms.Sequence(electronGsfTracking*conversionTrackSequence*allConversionSequence)
egammareco = cms.Sequence(electronSequence*conversionSequence*photonSequence)
egammaHighLevelRecoPrePF = cms.Sequence(gsfEcalDrivenElectronSequence*uncleanedOnlyElectronSequence*conversionSequence*photonSequence)

#egammaHighLevelRecoPostPF = cms.Sequence(gsfElectronMergingSequence*interestingEgammaIsoDetIds*photonIDSequence*eIdSequence*hfEMClusteringSequence)
#adding new gedGsfElectronSequence and gedPhotonSequence :
egammaHighLevelRecoPostPF = cms.Sequence(gsfElectronMergingSequence*gedGsfElectronSequence*interestingEgammaIsoDetIds*gedPhotonSequence*photonIDSequence*eIdSequence*hfEMClusteringSequence)


egammarecoFull = cms.Sequence(egammareco*interestingEgammaIsoDetIds*photonIDSequence*eIdSequence*hfEMClusteringSequence)
egammarecoWithID = cms.Sequence(egammareco*photonIDSequence*eIdSequence)
egammareco_woConvPhotons = cms.Sequence(electronSequence*photonSequence)
egammareco_withIsolation = cms.Sequence(egammareco*egammaIsolationSequence)
egammareco_withIsolation_woConvPhotons = cms.Sequence(egammareco_woConvPhotons*egammaIsolationSequence)
egammareco_withPhotonID = cms.Sequence(egammareco*photonIDSequence)
egammareco_withElectronID = cms.Sequence(egammareco*eIdSequence)

egammarecoFull_woHFElectrons = cms.Sequence(egammareco*interestingEgammaIsoDetIds*photonIDSequence*eIdSequence)



