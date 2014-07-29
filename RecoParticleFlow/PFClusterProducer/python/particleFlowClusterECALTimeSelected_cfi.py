import FWCore.ParameterSet.Config as cms
particleFlowClusterECALTimeSelected = cms.EDProducer(
    "PFClusterTimeSelector",
    src = cms.InputTag('particleFlowClusterECALWithTimeUncorrected'),

    cuts = cms.VPSet(
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(0.0),
            maxEnergy = cms.double(1.0),
            endcap = cms.bool(False),
            minTime = cms.double(-12.),
            maxTime = cms.double(12.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(0.0),
            maxEnergy = cms.double(1.0),
            endcap = cms.bool(True),
            minTime = cms.double(-31.5),
            maxTime = cms.double(31.5)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(2.0),
            endcap = cms.bool(False),
            minTime = cms.double(-6.),
            maxTime = cms.double(6.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(2.0),
            endcap = cms.bool(True),
            minTime = cms.double(-20.5),
            maxTime = cms.double(20.5)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(2.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(False),
            minTime = cms.double(-4.),
            maxTime = cms.double(4.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(2.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(True),
            minTime = cms.double(-12.),
            maxTime = cms.double(12.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(5.0),
            maxEnergy = cms.double(20.0),
            endcap = cms.bool(False),
            minTime = cms.double(-4.),
            maxTime = cms.double(4.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(5.0),
            maxEnergy = cms.double(20.0),
            endcap = cms.bool(True),
            minTime = cms.double(-5.),
            maxTime = cms.double(5.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(20.0),
            maxEnergy = cms.double(1e24),
            endcap = cms.bool(False),
            minTime = cms.double(-4.),
            maxTime = cms.double(4.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(20.0),
            maxEnergy = cms.double(1e24),
            endcap = cms.bool(True),
            minTime = cms.double(-5.),
            maxTime = cms.double(5.)
        )
   )
)

    
