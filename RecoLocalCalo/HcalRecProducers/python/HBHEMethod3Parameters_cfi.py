import FWCore.ParameterSet.Config as cms

# Configuration parameters for Method 3
m3Parameters = cms.PSet(
    applyTimeSlewM3         = cms.bool(True),
    pedestalUpperLimit      = cms.double(2.7),
    timeSlewParsType        = cms.int32(1),     # 0: TestStand, 1:Data, 2:MC, 3:InputPars. Parametrization function is par0 + par1*log(fC+par2).
    timeSlewPars            = cms.vdouble(12.2999, -2.19142, 0, 12.2999, -2.19142, 0, 12.2999, -2.19142, 0), # HB par0, HB par1, HB par2, BE par0, BE par1, BE par2, HE par0, HE par1, HE par2
    respCorrM3              = cms.double(0.95) # This factor is used to align the the Method3 with the Method2 response
)
