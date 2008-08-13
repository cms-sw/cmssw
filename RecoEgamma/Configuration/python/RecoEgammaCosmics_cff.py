import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.cosmicElectronSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.cosmicPhotonSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.cosmicConversionSequence_cff import *
#from RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff import *
#from RecoEgamma.PhotonIdentification.photonId_cff import *
#from RecoEgamma.ElectronIdentification.electronIdSequence_cff import *

egammarecoCosmics_woConvPhotons = cms.Sequence(cosmicElectronSequence*cosmicPhotonSequence)
egammarecoCosmics = cms.Sequence(cosmicElectronSequence*cosmicConversionSequence*cosmicPhotonSequence)
