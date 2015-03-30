# Misc loads for VID framework
from RecoMuon.MuonIdentification.muoMuonIDs_cfi import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

# Load the producer module to build full 5x5 cluster shapes and whatever 
# else is needed for IDs
#from RecoEgamma.ElectronIdentification.ElectronIDValueMapProducer_cfi import *

muoMuonIDSequence = cms.Sequence(muoMuonIDs)
