
import FWCore.ParameterSet.Config as cms

def configureEcalLocal25ns(process):
    process.ecalMultiFitUncalibRecHit.activeBXs = cms.vint32(-5,-4,-3,-2,-1,0,1,2,3,4),
    process.ecalMultiFitUncalibRecHit.useLumiInfoRunHeader = False
    return process

def configureEcalLocal50ns(process):
    process.ecalMultiFitUncalibRecHit.activeBXs = cms.vint32(-4,-2,0,2,4)
    process.ecalMultiFitUncalibRecHit.useLumiInfoRunHeader = False
    return process
  
def configureEcalLocalNoOOTPU(process):
    process.ecalMultiFitUncalibRecHit.activeBXs = cms.vint32(0)
    process.ecalMultiFitUncalibRecHit.useLumiInfoRunHeader = False
    return process
