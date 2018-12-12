from RecoEgamma.PhotonIdentification.photonIDValueMapProducer_cfi import *

import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(photonIDValueMapProducer, 
    esReducedRecHitCollection = "",
    esReducedRecHitCollectionMiniAOD = "",
)
