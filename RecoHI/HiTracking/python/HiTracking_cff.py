from RecoHI.HiTracking.hiMergedConformalPixelTracking_cff import *
from RecoHI.HiTracking.HIInitialJetCoreClusterSplitting_cff import *
from RecoHI.HiTracking.LowPtTracking_PbPb_cff import *
from RecoHI.HiTracking.hiLowPtTripletStep_cff import *
from RecoHI.HiTracking.hiMixedTripletStep_cff import *
from RecoHI.HiTracking.hiPixelPairStep_cff import *
from RecoHI.HiTracking.hiPixelLessStep_cff import *
from RecoHI.HiTracking.hiTobTecStep_cff import *
from RecoHI.HiTracking.hiDetachedTripletStep_cff import *
from RecoHI.HiTracking.hiJetCoreRegionalStep_cff import *
from RecoHI.HiTracking.MergeTrackCollectionsHI_cff import *
from RecoHI.HiTracking.hiLowPtQuadStep_cff import *
from RecoHI.HiTracking.hiHighPtTripletStep_cff import *
from RecoHI.HiTracking.hiDetachedQuadStep_cff import *


from RecoHI.HiMuonAlgos.hiMuonIterativeTk_cff import *

hiJetsForCoreTracking.cut = cms.string("pt > 100 && abs(eta) < 2.4")
hiJetCoreRegionalStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = cms.double( 10. )
hiJetCoreRegionalStepTrajectoryFilter.minPt = 10.0
siPixelClusters.ptMin = cms.double(100)
siPixelClusters.deltaRmax = cms.double(0.1)

from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
trackExtrapolator.trackSrc = cms.InputTag("hiGeneralTracks")

hiTracking_noRegitMuTask = cms.Task(
    hiBasicTrackingTask
    ,hiDetachedTripletStepTask
    ,hiLowPtTripletStepTask
    ,hiPixelPairStepTask
    )

hiTracking_noRegitMu_wSplittingTask = cms.Task(
    hiInitialJetCoreClusterSplittingTask
    ,hiBasicTrackingTask
    ,hiDetachedTripletStepTask
    ,hiLowPtTripletStepTask
    ,hiPixelPairStepTask
    )

hiTracking_noRegitMu_wSplitting_Phase1Task = cms.Task(
    hiInitialJetCoreClusterSplittingTask
    ,hiBasicTrackingTask
    ,hiLowPtQuadStepTask#New iteration
    ,hiHighPtTripletStepTask#New iteration
    ,hiDetachedQuadStepTask#New iteration
    ,hiDetachedTripletStepTask
    ,hiLowPtTripletStepTask
    ,hiPixelPairStepTask #no CA seeding implemented
    ,hiMixedTripletStepTask # large impact parameter tracks
    ,hiPixelLessStepTask    # large impact parameter tracks
    ,hiTobTecStepTask       # large impact parameter tracks
    )

hiTrackingTask = cms.Task(
    hiTracking_noRegitMuTask
    ,hiRegitMuTrackingAndStaTask
    ,hiGeneralTracks
    ,bestFinalHiVertexTask
    ,trackExtrapolator
    )
hiTracking = cms.Sequence(hiTrackingTask)

hiTracking_wSplittingTask = cms.Task(
    hiTracking_noRegitMu_wSplittingTask
    ,hiJetCoreRegionalStepTask 
    ,hiRegitMuTrackingAndStaTask
    ,hiGeneralTracks
    ,bestFinalHiVertexTask
    ,trackExtrapolator
    )
hiTracking_wSplitting = cms.Sequence(hiTracking_wSplittingTask)

hiTracking_wSplitting_Phase1Task = cms.Task(
    hiTracking_noRegitMu_wSplitting_Phase1Task
    ,hiJetCoreRegionalStepTask 
    ,hiRegitMuTrackingAndStaTask
    ,hiGeneralTracks
    ,bestFinalHiVertexTask
    ,trackExtrapolator
    )
hiTracking_wSplitting_Phase1 = cms.Sequence(hiTracking_wSplitting_Phase1Task)

hiTracking_wConformalPixelTask = cms.Task(
    hiTrackingTask
    ,hiMergedConformalPixelTrackingTask 
    )
hiTracking_wConformalPixel = cms.Sequence(hiTracking_wConformalPixelTask)
