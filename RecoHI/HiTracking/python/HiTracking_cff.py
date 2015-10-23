from RecoHI.HiTracking.hiMergedConformalPixelTracking_cff import *
from RecoHI.HiTracking.HIInitialJetCoreClusterSplitting_cff import *
from RecoHI.HiTracking.LowPtTracking_PbPb_cff import *
from RecoHI.HiTracking.hiLowPtTripletStep_cff import *
from RecoHI.HiTracking.hiMixedTripletStep_cff import *
from RecoHI.HiTracking.hiPixelPairStep_cff import *
from RecoHI.HiTracking.hiDetachedTripletStep_cff import *
from RecoHI.HiTracking.hiJetCoreRegionalStep_cff import *
from RecoHI.HiTracking.MergeTrackCollectionsHI_cff import *

from RecoHI.HiMuonAlgos.hiMuonIterativeTk_cff import *

hiJetsForCoreTracking.cut = cms.string("pt > 100 && abs(eta) < 2.4")
hiJetCoreRegionalStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = cms.double( 10. )
hiJetCoreRegionalStepTrajectoryFilter.minPt = 10.0
siPixelClusters.ptMin = cms.double(100)
siPixelClusters.deltaRmax = cms.double(0.1)

hiTracking_noRegitMu = cms.Sequence(
    hiBasicTracking
    *hiDetachedTripletStep
    *hiLowPtTripletStep
    *hiPixelPairStep
    )

hiTracking_noRegitMu_wSplitting = cms.Sequence(
    hiInitialJetCoreClusterSplitting
    *hiBasicTracking
    *hiDetachedTripletStep
    *hiLowPtTripletStep
    *hiPixelPairStep
    )

hiTracking = cms.Sequence(
    hiTracking_noRegitMu
    *hiRegitMuTrackingAndSta
    *hiGeneralTracks
    )

hiTracking_wSplitting = cms.Sequence(
    hiTracking_noRegitMu_wSplitting
    *hiJetCoreRegionalStep 
    *hiRegitMuTrackingAndSta
    *hiGeneralTracks
    )

hiTracking_wConformalPixel = cms.Sequence(
    hiTracking
    *hiMergedConformalPixelTracking 
    )
