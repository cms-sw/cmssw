
from RecoHI.HiTracking.HILowPtConformalPixelTracks_cfi import *
from RecoHI.HiTracking.LowPtTracking_PbPb_cff import *
from RecoHI.HiTracking.hiLowPtTripletStep_cff import *
from RecoHI.HiTracking.hiMixedTripletStep_cff import *
from RecoHI.HiTracking.hiPixelPairStep_cff import *
from RecoHI.HiTracking.hiDetachedTripletStep_cff import *
from RecoHI.HiTracking.MergeTrackCollectionsHI_cff import *

from RecoHI.HiMuonAlgos.hiMuonIterativeTk_cff import *

hiTracking_noRegitMu = cms.Sequence(
    hiBasicTracking
    *hiDetachedTripletStep
    *hiLowPtTripletStep
    *hiPixelPairStep
    )

hiTracking = cms.Sequence(
    hiTracking_noRegitMu
    *hiRegitMuTrackingAndSta
    *hiGeneralTracks
    )

hiTracking_wConformalPixel = cms.Sequence(
    hiBasicTracking
    *hiDetachedTripletStep
    *hiLowPtTripletStep
    *hiPixelPairStep
    *hiGeneralTracks
    *hiConformalPixelTracks    
    )

