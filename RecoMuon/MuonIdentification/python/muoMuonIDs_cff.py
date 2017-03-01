# Misc loads for VID framework
from RecoMuon.MuonIdentification.muoMuonIDs_cfi import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

muoMuonIDSequence = cms.Sequence(muoMuonIDs)
