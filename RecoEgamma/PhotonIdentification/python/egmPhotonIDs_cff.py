# Misc loads for VID framework
from RecoEgamma.PhotonIdentification.egmPhotonIDs_cfi import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Load the producer module to build full 5x5 cluster shapes and whatever 
# else is needed for IDs
from RecoEgamma.PhotonIdentification.PhotonIDValueMapProducer_cfi import *

# Load the producer for MVA IDs. Make sure it is also added to the sequence!
from RecoEgamma.PhotonIdentification.PhotonMVAValueMapProducer_cfi import *
from RecoEgamma.PhotonIdentification.PhotonRegressionValueMapProducer_cfi import *

# Load sequences for isolations computed with CITK for both AOD and miniAOD cases
from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationAOD_cff     import egmPhotonIsolationAODSequence
from RecoEgamma.EgammaIsolationAlgos.egmPhotonIsolationMiniAOD_cff import egmPhotonIsolationMiniAODSequence

# The exact sequence below is important. The MVA ValueMapProducer
# needs to be downstream from the ID ValueMapProducer because it relies 
# on some of its products, for example.

# The sequences for AOD and miniAOD are defined separately.
egmPhotonIDSequenceAOD = cms.Sequence(egmPhotonIsolationAODSequence * photonIDValueMapProducer * photonMVAValueMapProducer * egmPhotonIDs * photonRegressionValueMapProducer )
egmPhotonIDSequenceMiniAOD = cms.Sequence(egmPhotonIsolationMiniAODSequence * photonIDValueMapProducer * photonMVAValueMapProducer * egmPhotonIDs * photonRegressionValueMapProducer )

# The default case is miniAOD, however this can be controlled 
# via the data format argument of the function in vid_tools.py that switches
# on VID tools: switchOnVIDPhotonIdProducer(process, dataFormat)
egmPhotonIDSequence = cms.Sequence(egmPhotonIDSequenceMiniAOD)
