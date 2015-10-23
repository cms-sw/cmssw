import FWCore.ParameterSet.Config as cms


  
EmptySector = cms.PSet(
    # Give names to individual sectors
    name = cms.string("default"),
    # empty is equal to all possibilities (means no selections)
    rawId        = cms.vuint32(),
    subdetId     = cms.vuint32(),      #(1,2,3,4,5,6)
    layer        = cms.vuint32(),      #(1,2,3,4,5,6,7,8,9)
    side         = cms.vuint32(),      #(1,2)
    half         = cms.vuint32(),      #(1,2)
    rod          = cms.vuint32(),      #(1,...74)
    ring         = cms.vuint32(),      #(1,2,3,4,5,6,7)
    petal        = cms.vuint32(),      #(1,2,3,4,5,6,7,8)
    blade        = cms.vuint32(),      #(1,...24)
    panel        = cms.vuint32(),      #(1,2)
    outerInner   = cms.vuint32(),      #(1,2)
    module       = cms.vuint32(),      #(1,...20)
    rodAl        = cms.vuint32(),      #(1,...28)
    bladeAl      = cms.vuint32(),      #(1,...12)
    nStrips      = cms.vuint32(),      #(512,768)
    isDoubleSide = cms.vuint32(),      #(1,2)  1: only virtual combined DS module, 2: only physical modules // DoubleSide means Combined virtual Module, so one entry for double-sided module -> exclude always (now already excluded in trackerTreeGenerator)
    isRPhi       = cms.vuint32(),      #(1,2)  1: only RPhi, 2: only Stereo  // RPhi means also all single-sided modules (in every layer - e.g. TIB Layer 3,4)
    uDirection   = cms.vint32(),       #(-1,1)
    vDirection   = cms.vint32(),       #(-1,1)
    wDirection   = cms.vint32(),       #(-1,1)
    posR         = cms.vdouble(),      #(0.,120.) must contain Intervals (even nr of arguments)
    posPhi       = cms.vdouble(),      #(-3.5,3.5) must contain Intervals
    posEta       = cms.vdouble(),      #(-3.,3.) must contain Intervals
    posX         = cms.vdouble(),      #(-120.,120.) must contain Intervals
    posY         = cms.vdouble(),      #(-120.,120.) must contain Intervals
    posZ         = cms.vdouble()       #(-280.,280.) must contain Intervals
)
