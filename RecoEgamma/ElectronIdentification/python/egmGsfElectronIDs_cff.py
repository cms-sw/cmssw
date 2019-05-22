# Misc loads for VID framework
from RecoEgamma.ElectronIdentification.egmGsfElectronIDs_cfi import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Load the producer module to build full 5x5 cluster shapes and whatever
# else is needed for IDs
# NOTE: Presently this producer is not needed because all variables
# that it produces are already available as standard electron variables.
# When it is needed again, uncomment the lone below.
#   Do not forget to also add "electronIDValueMapProducer" to the sequence
# defined below!
#
#from RecoEgamma.ElectronIdentification.ElectronIDValueMapProducer_cfi import *

# Load the producer for MVA IDs. Make sure it is also added to the sequence!
from RecoEgamma.ElectronIdentification.ElectronMVAValueMapProducer_cfi import *

egmGsfElectronIDTask = cms.Task(
    electronMVAValueMapProducer,
    egmGsfElectronIDs
)
egmGsfElectronIDSequence = cms.Sequence(egmGsfElectronIDTask)
