import FWCore.ParameterSet.Config as cms

HSParameterBlock = cms.PSet(
    HSParameters = cms.PSet(
        nTRsteps = cms.int32(40),
        lossesOpt = cms.int32(0),
        depthStep = cms.double(0.5),
        balanceEH = cms.double(0.9),
        eSpotSize = cms.double(0.2),
        hcalDepthFactor = cms.double(1.1),
        transRparam = cms.double(1.0),
        nDepthSteps = cms.int32(10),
        maxTRfactor = cms.double(4.0),
        criticalHDEnergy = cms.double(3.0)
        )
    )

