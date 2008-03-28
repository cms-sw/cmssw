import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.electronSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.photonSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.conversionSequence_cff import *
egammareco = cms.Sequence(electronSequence*conversionSequence*photonSequence)
egammareco_woConvPhotons = cms.Sequence(electronSequence*photonSequence)

