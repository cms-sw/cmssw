import FWCore.ParameterSet.Config as cms

# Configuration parameters for Method 3
m3Parameters = cms.PSet(
    applyTimeSlewM3         = cms.bool(True),
    timeSlewParsType        = cms.int32(3),     # 0: TestStand, 1:Data, 2:MC, 3:HBHE. Parametrization function is par0 + par1*log(fC+par2).
    respCorrM3              = cms.double(1.0)   # This factor is used to align the Method3 with the Method2 response
)
