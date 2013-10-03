from RecoHI.HiTracking.LowPtTracking_PbPb_cff import *
from RecoHI.HiTracking.hiSecondPixelTripletStep_cff import *
from RecoHI.HiTracking.hiMixedTripletStep_cff import *
from RecoHI.HiTracking.hiPixelPairStep_cff import *
from RecoHI.HiTracking.MergeTrackCollectionsHI_cff import *

hiIterTracking = cms.Sequence(
    heavyIonTracking
    *hiSecondPixelTripletStep
    *hiPixelPairStep
    *hiGeneralTracks
    )
