import FWCore.ParameterSet.Config as cms

### Set scaling factors

fieldScaling = cms.PSet(
  scalingVolumes = cms.vint32(
    # Barrel yoke plates -- all sectors
    14100,14200,  #TC
    17600,        #YB-2/1
    17800,17900,  #YB-1/1
    18100,        #YB0/1
    18300,18400,  #YB+1/1
    18600,        #YB+2/1
    23100,        #YB-2/2
    23300,23400,  #YB-1/2
    23600,        #YB0/2
    23800,23900,  #YB+1/2
    24100,        #YB+2/2
    28600,        #YB-2/3
    28800,28900,  #YB-1/3
    29100,        #YB0/3
    29300,29400,  #YB+1/3
    29600,        #YB+2/3
    # Barrel yoke plates -- sector 9, layer 3
    28609,        #YB-2/3
    28809,28909,  #YB-1/3
    29109,        #YB0/3
    29309,29409,  #YB+1/3
    29609,        #YB+2/3
    # Barrel yoke plates -- sector 10, layer 3
    28610,        #YB-2/3
    28810,28910,  #YB-1/3
    29110,        #YB0/3
    29310,29410,  #YB+1/3
    29610,        #YB+2/3
    # Barrel yoke plates -- sector 11, layer 3
    28611,        #YB-2/3
    28811,28911,  #YB-1/3
    29111,        #YB0/3
    29311,29411,  #YB+1/3
    29611         #YB+2/3
    ),

scalingFactors = cms.vdouble(
    # Barrel yoke plates
    1,1, #TC
    0.994,        #YB-2/1
    1.004,1.004 , #YB-1/1
    1.005,        #YB0/1
    1.004,1.004,  #YB+1/1
    0.994,        #YB+2/1
    0.965,        #YB-2/2
    0.958,0.958,  #YB-1/2
    0.953,        #YB0/2
    0.958,0.958,  #YB+1/2
    0.965,        #YB+2/2
    0.918,        #YB-2/3
    0.924,0.924,  #YB-1/3
    0.906,        #YB0/3
    0.924,0.924,  #YB+1/3
    0.918,        #YB+2/3
    # Barrel yoke plates -- sector 9, layer 3
    0.991,        #YB-2/3
    0.998,0.998,  #YB-1/3
    0.978,        #YB0/3
    0.998,0.998,  #YB+1/3
    0.991,        #YB+2/3
    # Barrel yoke plates -- sector 10, layer 3
    0.991,        #YB-2/3
    0.998,0.998,  #YB-1/3
    0.978,        #YB0/3
    0.998,0.998,  #YB+1/3
    0.991,        #YB+2/3
    # Barrel yoke plates -- sector 11, layer 3
    0.991,        #YB-2/3
    0.998,0.998,  #YB-1/3
    0.978,        #YB0/3
    0.998,0.998,  #YB+1/3
    0.991         #YB+2/3
    )
)
