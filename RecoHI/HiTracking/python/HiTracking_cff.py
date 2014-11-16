
from RecoHI.HiTracking.HILowPtConformalPixelTracks_cfi import *
from RecoHI.HiTracking.LowPtTracking_PbPb_cff import *
from RecoHI.HiTracking.hiSecondPixelTripletStep_cff import *
from RecoHI.HiTracking.hiMixedTripletStep_cff import *
from RecoHI.HiTracking.hiPixelPairStep_cff import *
from RecoHI.HiTracking.MergeTrackCollectionsHI_cff import *

from RecoHI.HiMuonAlgos.hiMuonIterativeTk_cff import *

hiTracking_noRegitMu = cms.Sequence(
    hiBasicTracking
    *hiSecondPixelTripletStep
    *hiPixelPairStep
    )

hiTracking = cms.Sequence(
    hiTracking_noRegitMu
    *hiRegitMuTrackingAndSta
    *hiGeneralTracks
    )

hiTracking_wConformalPixel = cms.Sequence(
    hiBasicTracking
    *hiSecondPixelTripletStep
    *hiPixelPairStep
    *hiGeneralTracks
    *hiConformalPixelTracks    
    )

