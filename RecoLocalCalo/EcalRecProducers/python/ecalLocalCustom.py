import FWCore.ParameterSet.Config as cms

def configureEcalLocal25ns(process):
    process.ecalMultiFitUncalibRecHit.cpu.activeBXs = [-5,-4,-3,-2,-1,0,1,2,3,4],
    process.ecalMultiFitUncalibRecHit.cpu.useLumiInfoRunHeader = False
    return process

def configureEcalLocal50ns(process):
    process.ecalMultiFitUncalibRecHit.cpu.activeBXs = [-4,-2,0,2,4]
    process.ecalMultiFitUncalibRecHit.cpu.useLumiInfoRunHeader = False
    return process
  
def configureEcalLocalNoOOTPU(process):
    process.ecalMultiFitUncalibRecHit.cpu.activeBXs = [0]
    process.ecalMultiFitUncalibRecHit.cpu.useLumiInfoRunHeader = False
    return process
