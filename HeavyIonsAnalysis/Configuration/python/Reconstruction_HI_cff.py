import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.Configuration.HighPtTracking_PbPb_cff import *
# from HeavyIonsAnalysis.Configuration.LowPtTracking_PbPb_cff import *
# from HeavyIonsAnalysis.Configuration.Tracking_PbPb_cff import *
# from HeavyIonsAnalysis.Configuration.HeavyIonTracking_PbPb_cff import *

from RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *
from RecoEcal.EgammaClusterProducers.islandClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.hybridClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.multi5x5ClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.multi5x5PreshowerClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.preshowerClusteringSequence_cff import *
from RecoLocalCalo.Configuration.hcalLocalReco_cff import *
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
from RecoMuon.Configuration.RecoMuon_cff import *
from HeavyIonsAnalysis.Configuration.IterativeConePu5Jets_PbPb_cff import *
from RecoHI.HiCentralityAlgos.HiCentrality_cfi import *
from RecoHI.HiEvtPlaneAlgos.HiEvtPlane_cfi import *
# ECAL Clustering Algorithm
# Added Multi5x5 algorithm into the reconstruction by Yen-Jie
ecalloc = cms.Sequence(ecalWeightUncalibRecHit*ecalRecHit*ecalPreshowerRecHit)
ecalcst = cms.Sequence(islandClusteringSequence*hybridClusteringSequence*preshowerClusteringSequence*multi5x5ClusteringSequence*multi5x5PreshowerClusteringSequence)
caloReco = cms.Sequence(ecalloc*ecalcst*hcalLocalRecoSequence)

reconstruct_PbPb = cms.Sequence(hiTrackingWithOfflineBeamSpot*caloReco*runjets*hiCentrality*hiEvtPlane)


