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
    29600         #YB+2/3
    ),

scalingFactors = cms.vdouble(
    # Barrel yoke plates
    1,1,               #TC
    0.935517,          #YB-2/1
    0.956464,0.956464, #YB-1/1
    0.935383,          #YB0/1
    0.956464,0.956464, #YB+1/1
    0.935517,          #YB+2/1
    0.981935,          #YB-2/2
    0.949556,0.949556, #YB-1/2
    0.961469,          #YB0/2
    0.949556,0.949556, #YB+1/2
    0.981935,          #YB+2/2
    0.862724,          #YB-2/3
    0.921487,0.921487, #YB-1/3
    0.912362,          #YB0/3
    0.921487,0.921487, #YB+1/3
    0.862724           #YB+2/3
    )
)
