# Misc loads for VID framework
from RecoEgamma.PhotonIdentification.egmPhotonIDs_cfi import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Load the producer module to build full 5x5 cluster shapes and whatever 
# else is needed for IDs
from RecoEgamma.PhotonIdentification.PhotonIDValueMapProducer_cfi import *

# Load the producer for MVA IDs. Make sure it is also added to the sequence!
from RecoEgamma.PhotonIdentification.PhotonMVAValueMapProducer_cfi import *
from RecoEgamma.PhotonIdentification.PhotonRegressionValueMapProducer_cfi import *

# The sequence below is important. The MVA ValueMapProducer
# needs to be downstream from the ID ValueMapProducer because it relies 
# on some of its products
egmPhotonIDSequence = cms.Sequence(photonIDValueMapProducer * photonMVAValueMapProducer * egmPhotonIDs * photonRegressionValueMapProducer )
