import FWCore.ParameterSet.Config as cms

# from HeavyIonsAnalysis.Configuration.HighPtTracking_PbPb_cff import *
# from HeavyIonsAnalysis.Configuration.LowPtTracking_PbPb_cff import *

from HeavyIonsAnalysis.Configuration.Tracking_PbPb_cff import *

from RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *
from RecoEcal.EgammaClusterProducers.islandClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.hybridClusteringSequence_cff import *
from RecoEcal.EgammaClusterProducers.preshowerClusteringSequence_cff import *
from RecoLocalCalo.Configuration.hcalLocalReco_cff import *
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
from RecoMuon.Configuration.RecoMuon_cff import *
from HeavyIonsAnalysis.Configuration.IterativeConePu5Jets_PbPb_cff import *
from RecoHI.HiCentralityAlgos.HiCentrality_cfi import *
from RecoHI.HiEvtPlaneAlgos.HiEvtPlane_cfi import *
ecalloc = cms.Sequence(ecalWeightUncalibRecHit*ecalRecHit*ecalPreshowerRecHit)
ecalcst = cms.Sequence(islandClusteringSequence*hybridClusteringSequence*preshowerClusteringSequence)
caloReco = cms.Sequence(ecalloc*ecalcst*hcalLocalRecoSequence)
reconstruct_PbPb = cms.Sequence(hiTrackingWithOfflineBeamSpot*caloReco*runjets*hiCentrality*hiEvtPlane)


