import FWCore.ParameterSet.Config as cms

### Scaling factors obtained using CRAFT08 + CRAFT09 data on top of version 090601.

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
    29600         #YB+2/3
    ),

scalingFactors = cms.vdouble(
    # Barrel yoke plates
    1,1,               #TC
    0.945721,          #YB-2/1
    0.958026,0.958026, #YB-1/1
    0.933539,          #YB0/1
    0.958026,0.958026, #YB+1/1
    0.945721,          #YB+2/1
    0.987094,          #YB-2/2
    0.952123,0.952123, #YB-1/2
    0.964203,          #YB0/2
    0.952123,0.952123, #YB+1/2
    0.987094,          #YB+2/2
    0.871933,          #YB-2/3
    0.923347,0.923347, #YB-1/3
    0.913604,          #YB0/3
    0.923347,0.923347, #YB+1/3
    0.871933           #YB+2/3
    )
)
