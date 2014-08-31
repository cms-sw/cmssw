# Misc loads for VID framework
from EgammaAnalysis.ElectronTools.egmGsfElectronIDs_cfi import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Load the producer module to build full 5x5 cluster shapes and whatever 
# else is needed for IDs
from EgammaAnalysis.ElectronTools.ElectronIDValueMapProducer_cfi import *

egmGsfElectronIDSequence = cms.Sequence(electronIDValueMapProducer * egmGsfElectronIDs)
