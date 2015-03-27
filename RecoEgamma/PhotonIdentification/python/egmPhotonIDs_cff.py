# Misc loads for VID framework
from RecoEgamma.PhotonIdentification.egmPhotonIDs_cfi import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Load the producer module to build full 5x5 cluster shapes and whatever 
# else is needed for IDs
from RecoEgamma.PhotonIdentification.PhotonIDValueMapProducer_cfi import *

egmPhotonIDSequence = cms.Sequence(photonIDValueMapProducer * egmPhotonIDs)
