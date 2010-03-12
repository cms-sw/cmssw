import FWCore.ParameterSet.Config as cms

def customise(process):
    ## preshower baseline substraction is done already in data.
    process.ecalPreshowerRecHit.ESBaseline = cms.int32(0) 
    process.ecalPreshowerRecHit.ESRecoAlgo = cms.untracked.int32(1)

    return (process)
