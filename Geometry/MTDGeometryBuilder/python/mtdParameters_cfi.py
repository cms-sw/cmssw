import FWCore.ParameterSet.Config as cms

from Geometry.MTDGeometryBuilder.mtdParametersBase_cfi import mtdParametersBase

mtdParameters = mtdParametersBase.clone()
del mtdParametersBase

mtdParameters.vpars = cms.vint32(4,4,4,24)
mtdParameters.vitems = cms.VPSet( 
    cms.PSet(  #BTL
        subdetPars = cms.vint32(22,24,16,10,
                                0x1,0x3,0x3F,0x3F,
                                4,4,4,1) #rows / columns / nROCs x / nROCs y
        ), 
    cms.PSet(  #ETL
        subdetPars = cms.vint32(22,24,16,7,
                                0x1,0x3,0x3F,0xFF,
                                24,4,2,8) #rows / columns / nROCs x / nROCs y
        )
    )
