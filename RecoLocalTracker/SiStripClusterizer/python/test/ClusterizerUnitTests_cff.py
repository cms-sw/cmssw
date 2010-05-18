import FWCore.ParameterSet.Config as cms    

from RecoLocalTracker.SiStripClusterizer.test.ClusterizerUnitTestFunctions_cff import *

clusterizerTests = ClusterizerTest( "Default Clusterizer Settings",
                                    cms.PSet( Algorithm = cms.string("ThreeThresholdAlgorithm"),
                                              ChannelThreshold = cms.double(2),
                                              SeedThreshold    = cms.double(3),
                                              ClusterThreshold = cms.double(5),
                                              MaxSequentialHoles = cms.uint32(0),
                                              MaxSequentialBad   = cms.uint32(1),
                                    MaxAdjacentBad     = cms.uint32(0),
                                              QualityLabel = cms.string("")
                                    ),
                                    [
    DetUnit( "[] = []",
             [  ],
             [ #none
               ] ),
    DetUnit( "(4/1) = []",
             [ digi(  10, 4,  noise1, gain1, good)  ],
             [ #none
               ] ),
    DetUnit( "(5/1) = [5]",
             [ digi(    10, 5,  noise1, gain1, good)  ],
             [ cluster( 10, [5])
               ] ),
    DetUnit( "(110/1) = [110]",
             [ digi(    10, 110,  noise1, gain1, good)  ],
             [ cluster( 10, [110])
               ] ),
    DetUnit( "(24/5) = []",
             [ digi(  10, 24,  5*noise1, gain1, good)  ],
             [ #none
               ] ),
    DetUnit( "(25/5) = [25]",
             [ digi(    10, 25,  5*noise1, gain1, good)  ],
             [ cluster( 10, [25])
               ] ),
    DetUnit( "(111/5) = [111]",
             [ digi(    10, 111,  5*noise1, gain1, good)  ],
             [ cluster( 10, [111])
               ] ),
    DetUnit( "(25/5)(9/5) = [25]    <---------------|",
             [ digi(    10, 25,  5*noise1, gain1, good),
               digi(    11, 9,  5*noise1, gain1, good)  ],
             [ cluster( 10, [25])
               ] ),
    DetUnit( "(25/5)(10/5) =  []    <---------------| Strange",
             [ digi(    10, 25,  5*noise1, gain1, good),
               digi(    11, 10,  5*noise1, gain1, good)  ],
             [ #none
               ] ),
    DetUnit( "(25/5)(11/5) = [25,11]    <-----------|",
             [ digi(    10, 25,  5*noise1, gain1, good),
               digi(    11, 11,  5*noise1, gain1, good)  ],
             [ cluster(10, [25,11])
               ] ),
    DetUnit( "(7/1)(4/2) =  []  <------ Additional noise from neighbor nullifies good cluster! Reconsider!",
             [ digi(  10, 7,   noise1, gain1, good),
               digi(  11, 4,  2*noise1, gain1, good) ],
             [ #none
               ] ),
    DetUnit( "(3/1)(3/1) = []",
             [ digi(  10, 3,  noise1, gain1, good),
               digi(  11, 3,  noise1, gain1, good) ],
             [ #none
               ] ),
    DetUnit( "(2/1)(6/2) = []",
             [ digi(  10, 2,  noise1, gain1, good),
               digi(  11, 6,  2*noise1, gain1, good) ],
             [ #none
               ] ),
    DetUnit( "(16/5)(11/4) = []",
             [ digi(  10, 16,  5*noise1, gain1, good),
               digi(  11, 11,  5*noise1, gain1, good) ],
             [ #none
               ] ),
    DetUnit( "(2/1)(3/1)(2/1)(3/1)(2/1) = [2,3,2,3,2]",
             [ digi(  10, 2,  noise1, gain1, good),
               digi(  11, 3,  noise1, gain1, good),
               digi(  12, 2,  noise1, gain1, good),
               digi(  13, 3,  noise1, gain1, good),
               digi(  14, 2,  noise1, gain1, good) ],
             [ cluster(10,[2,3,2,3,2,])
               ] ),
    DetUnit( "(2/1)(3/1)(2/1)(3/1)(2/1)(2/2) = [2,3,2,3,2]",
             [ digi(  10, 2,  noise1, gain1, good),
               digi(  11, 3,  noise1, gain1, good),
               digi(  12, 2,  noise1, gain1, good),
               digi(  13, 3,  noise1, gain1, good),
               digi(  14, 2,  noise1, gain1, good),
               digi(  15, 2,  2*noise1, gain1, good) ],
             [ cluster(10,[2,3,2,3,2,])
               ] ),
    DetUnit( "(110/1)(100/1) = [110,100]",
             [ digi(  10, 110,  noise1, gain1, good),
               digi(  11, 100,  noise1, gain1, good) ],
             [ cluster(  10, [110, 100])
               ] ),
    DetUnit( "(110/1)_(100/1) = [110],[100]",
             [ digi(  10, 110,  noise1, gain1, good),
               digi(  11,   0,  noise1, gain1, good),
               digi(  12, 100,  noise1, gain1, good) ],
             [ cluster(  10, [110]),
               cluster(  12, [100])
               ] ),
    DetUnit( "Saturation at 254, gain has no effect",
             [ digi(  10, 254,  noise1, 1.3*gain1, good) ],
             [ cluster(  10, [254])
               ] ),
    DetUnit( "First strip saturated",
             [ digi(  10, 254,  noise1, 2*gain1, good),
               digi(  11, 100,  noise1, 2*gain1, good) ],
             [ cluster(  10, [254,50])
               ] ),
    DetUnit( "Last strip saturated",
             [ digi(  10, 100,  noise1, 2*gain1, good),
               digi(  11, 254,  noise1, 2*gain1, good) ],
             [ cluster(  10, [50,254])
               ] ),
    DetUnit( "Middle strip saturated",
             [ digi(  10, 100,  noise1, 2*gain1, good),
               digi(  11, 254,  noise1, 2*gain1, good),
               digi(  12, 100,  noise1, 2*gain1, good) ],
             [ cluster(  10, [50,254,50])
               ] ),
    DetUnit( "Saturation at 255, gain has no effect",
             [ digi(  10, 255,  noise1, 1.3*gain1, good) ],
             [ cluster(  10, [255])
               ] ),
    DetUnit( "Gain greater than 1",
             [ digi(  10, 110,  noise1, 1.3*gain1, good) ],
             [ cluster(  10, [85])
               ] ),
    DetUnit( "Gain less than 1",
             [ digi(  10, 110,  noise1, 0.82*gain1, good) ],
             [ cluster(  10, [134])
               ] ),
    DetUnit( "Gain less than 1 pushes charge above 1022",
             [ digi(  10, 253,  noise1, 0.2*gain1, good) ],
             [ cluster(  10, [255])
               ] ),
    DetUnit( "Gain less than 1 pushes charge above 255, but not above 1022",
             [ digi(  10, 253,  noise1, 0.9*gain1, good) ],
             [ cluster(  10, [254])
               ] ),
    DetUnit( "Gain less than 1 pushes charge above 511, but not above 1022",
             [ digi(  10, 253,  noise1, 0.4*gain1, good) ],
             [ cluster(  10, [254])
               ] ),
    DetUnit( "Two gains (apv boundary)",
             [ digi(  127, 110,  noise1, gain1, good),
               digi(  128, 110,  noise1, 1.1, good) ],
             [ cluster(  127, [110, 100])
               ] ),
    DetUnit( "Throws InvalidChargeException",
             [ digi(  19, 256,   noise1, gain1, good) ],
             [ 
               ],
             Invalid),
    DetUnit( "Left edge",
             [ digi(  0, 100,   noise1, gain1, good),],
             [ cluster(0,[100])
               ] ),
    DetUnit( "Right edge",
             [ digi(  767, 100,   noise1, gain1, good),],
             [ cluster(767,[100])
               ] ),
    DetUnit( "Left edge two strips",
             [ digi(  0, 100,   noise1, gain1, good),
               digi(  1, 100,   noise1, gain1, good),],
             [ cluster(0,[100,100])
               ] ),
    DetUnit( "Right edge two strips",
             [ digi(  766, 100,   noise1, gain1, good),
               digi(  767, 100,   noise1, gain1, good),],
             [ cluster(766,[100,100])
               ] ),
    DetUnit( "Wide cluster",
             [ digi(  10, 110,  noise1, gain1, good),
               digi(  11, 110,  noise1, gain1, good),
               digi(  12, 110,  noise1, gain1, good),
               digi(  13, 110,  noise1, gain1, good),
               digi(  14, 110,  noise1, gain1, good),
               digi(  15, 110,  noise1, gain1, good),
               digi(  16, 110,  noise1, gain1, good),
               digi(  17, 110,  noise1, gain1, good),
               digi(  18, 110,  noise1, gain1, good),
               digi(  19,  20,  noise1, gain1, good),
               digi(  20, 100,  noise1, gain1, good) ],
             [ cluster(  10, [110,110,110,110,110,110,110,110,110,20,100])
               ] ),
    DetUnit( "(110/1)(100/1) = [110,100]",
             [ digi(  10, 110,  noise1, gain1, good),
               digi(  11, 100,  noise1, gain1, good) ],
             [ cluster(  10, [110, 100])
               ] ),
    DetUnit( "(110/1)X = [110]",
             [ digi(  10, 110,  noise1, gain1, good),
               digi(  11, 110,  noise1, gain1,  bad) ],
             [ cluster(  10, [110])
               ] ),
    DetUnit( "X(110/1) = [110]",
             [ digi(  10, 110,  noise1, gain1, bad),
               digi(  11, 110,  noise1, gain1,  good) ],
             [ cluster(  11, [110])
               ] ),
    DetUnit( "XX(110/1) = [110]",
             [ digi(  9, 110,  noise1, gain1, bad),
               digi(  10, 110,  noise1, gain1, bad),
               digi(  11, 110,  noise1, gain1,  good) ],
             [ cluster(  11, [110])
               ] ),
    DetUnit( "(110/1)X(100/1) = [110,0,100]",
             [ digi(  10, 110,  noise1, gain1, good),
               digi(  11, 110,  noise1, gain1,  bad),
               digi(  12, 100,  noise1, gain1, good) ],
             [ cluster(  10, [110,0,100])
               ] ),
    DetUnit( "(110/1)x(100/1) = [110,0,100]",
             [ digi(  10, 110,  noise1, gain1, good),
               digi(  11,   0,  noise1, gain1,  bad),
               digi(  12, 100,  noise1, gain1, good) ],
             [ cluster(  10, [110, 0,100])
               ] ),
    DetUnit( "X(110/1)x(100/1)X = [110,0,100]",
             [ digi(  9, 110,  noise1, gain1, bad),
               digi(  10, 110,  noise1, gain1, good),
               digi(  11,   0,  noise1, gain1,  bad),
               digi(  12, 100,  noise1, gain1,  good),
               digi(  13, 100,  noise1, gain1, bad) ],
             [ cluster(  10, [110,0,100])
               ] ),
    DetUnit( "(110/1)_(100/1) = [110],[100]",
             [ digi(  10, 110,  noise1, gain1, good),
               digi(  11,   0,  noise1, gain1, good),
               digi(  12, 100,  noise1, gain1, good) ],
             [ cluster(  10, [110]),
               cluster(  12, [100])
               ] )
    ]
                                           )
