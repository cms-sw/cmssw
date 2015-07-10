# Misc loads for VID framework
from RecoEgamma.ElectronIdentification.egmGsfElectronIDs_cfi import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Load the producer module to build full 5x5 cluster shapes and whatever 
# else is needed for IDs
from RecoEgamma.ElectronIdentification.ElectronIDValueMapProducer_cfi import *

# Load the producer for MVA IDs. Make sure it is also added to the sequence!
from RecoEgamma.ElectronIdentification.ElectronMVAValueMapProducer_cfi import *

egmGsfElectronIDSequence = cms.Sequence( electronMVAValueMapProducer * egmGsfElectronIDs)
