
import FWCore.ParameterSet.Config as cms

def configureEcalLocal50ns(process):
    process.ecalMultiFitUncalibRecHit.activeBXs = cms.vint32(-4,-2,0,2,4)
    return process
  
def configureEcalLocalNoOOTPU(process):
    process.ecalMultiFitUncalibRecHit.activeBXs = cms.vint32(0)
    return process
