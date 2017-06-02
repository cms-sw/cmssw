### Currently, only performs regression: wait to do OOT "ID" at analysis level

from RecoEgamma.PhotonIdentification.PhotonRegressionValueMapProducer_cfi import *

ootPhotonRegressionValueMapProducer = photonRegressionValueMapProducer.clone()
ootPhotonRegressionValueMapProducer.src = cms.InputTag('ootPhotons') # AOD
ootPhotonRegressionValueMapProducer.srcMiniAOD = cms.InputTag('slimmedOOTPhotons') # miniAOD

egmOOTPhotonIDTask = cms.Task(
    ootPhotonRegressionValueMapProducer
)
egmOOTPhotonIDSequence = cms.Sequence(egmOOTPhotonIDTask)
