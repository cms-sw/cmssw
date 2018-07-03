# Misc loads for VID framework
from RecoMuon.MuonIdentification.muoMuonIDs_cfi import *
from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

muoMuonIDTask = cms.Task(muoMuonIDs)
muoMuonIDSequence = cms.Sequence(muoMuonIDTask)
