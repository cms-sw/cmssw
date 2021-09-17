from RecoEgamma.EgammaIsolationAlgos.interestingEgammaIsoDetIdsSequence_cff import *
from RecoEgamma.PhotonIdentification.photonId_cff import *
from RecoEgamma.ElectronIdentification.electronIdSequence_cff import *
from RecoEgamma.EgammaHFProducers.hfEMClusteringSequence_cff import *
from RecoEgamma.EgammaIsolationAlgos.egmIsolationDefinitions_cff import *


eidRobustLoose.verticesCollection = "hiSelectedVertex"
eidRobustTight.verticesCollection = "hiSelectedVertex"
eidRobustHighEnergy.verticesCollection = "hiSelectedVertex"
eidLoose.verticesCollection = "hiSelectedVertex"
eidTight.verticesCollection = "hiSelectedVertex"
hfRecoEcalCandidate.VertexCollection = "hiSelectedVertex"

egammaHighLevelRecoPostPFTask = cms.Task(interestingEgammaIsoDetIdsTask,egmIsolationTask,photonIDTask,photonIDTaskGED,eIdTask,hfEMClusteringTask)
egammaHighLevelRecoPostPF = cms.Sequence(egammaHighLevelRecoPostPFTask)
