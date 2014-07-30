import FWCore.ParameterSet.Config as cms
particleFlowClusterHBHETimeSelected = cms.EDProducer(
    "PFClusterTimeSelector",
    src = cms.InputTag('particleFlowClusterHBHE'),

    cuts = cms.VPSet(
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(0.0),
            maxEnergy = cms.double(1.0),
            endcap = cms.bool(False),
            minTime = cms.double(-22.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(0.0),
            maxEnergy = cms.double(1.0),
            endcap = cms.bool(True),
            minTime = cms.double(-22.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(0.0),
            maxEnergy = cms.double(1.0),
            endcap = cms.bool(False),
            minTime = cms.double(-22.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(0.0),
            maxEnergy = cms.double(1.0),
            endcap = cms.bool(True),
            minTime = cms.double(-22.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(0.0),
            maxEnergy = cms.double(1.0),
            endcap = cms.bool(False),
            minTime = cms.double(-22.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(0.0),
            maxEnergy = cms.double(1.0),
            endcap = cms.bool(True),
            minTime = cms.double(-22.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(False),
            minTime = cms.double(-20.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(True),
            minTime = cms.double(-20.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(False),
            minTime = cms.double(-20.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(True),
            minTime = cms.double(-20.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(False),
            minTime = cms.double(-20.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(True),
            minTime = cms.double(-20.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(5.),
            maxEnergy = cms.double(10.0),
            endcap = cms.bool(False),
            minTime = cms.double(-18.),
            maxTime = cms.double(10.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(5.0),
            maxEnergy = cms.double(10.0),
            endcap = cms.bool(True),
            minTime = cms.double(-18.),
            maxTime = cms.double(10.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(5.),
            maxEnergy = cms.double(10.0),
            endcap = cms.bool(False),
            minTime = cms.double(-18.),
            maxTime = cms.double(10.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(5.0),
            maxEnergy = cms.double(10.0),
            endcap = cms.bool(True),
            minTime = cms.double(-18.),
            maxTime = cms.double(10.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(5.),
            maxEnergy = cms.double(10.0),
            endcap = cms.bool(False),
            minTime = cms.double(-18.),
            maxTime = cms.double(10.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(5.0),
            maxEnergy = cms.double(10.0),
            endcap = cms.bool(True),
            minTime = cms.double(-18.),
            maxTime = cms.double(10.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(10.),
            maxEnergy = cms.double(50.0),
            endcap = cms.bool(False),
            minTime = cms.double(-14.),
            maxTime = cms.double(6.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(10.0),
            maxEnergy = cms.double(50.0),
            endcap = cms.bool(True),
            minTime = cms.double(-14.),
            maxTime = cms.double(6.)
        )

    )
)

    
