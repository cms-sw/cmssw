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
            minTime = cms.double(-30.),
            maxTime = cms.double(30.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(0.0),
            maxEnergy = cms.double(1.0),
            endcap = cms.bool(True),
            minTime = cms.double(-30.),
            maxTime = cms.double(30.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(0.0),
            maxEnergy = cms.double(1.0),
            endcap = cms.bool(False),
            minTime = cms.double(-30.),
            maxTime = cms.double(30.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(0.0),
            maxEnergy = cms.double(1.0),
            endcap = cms.bool(True),
            minTime = cms.double(-30.),
            maxTime = cms.double(30.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(0.0),
            maxEnergy = cms.double(1.0),
            endcap = cms.bool(False),
            minTime = cms.double(-30.),
            maxTime = cms.double(30.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(0.0),
            maxEnergy = cms.double(1.0),
            endcap = cms.bool(True),
            minTime = cms.double(-30.),
            maxTime = cms.double(30.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(2.0),
            endcap = cms.bool(False),
            minTime = cms.double(-20.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(2.0),
            endcap = cms.bool(True),
            minTime = cms.double(-20.),
            maxTime = cms.double(16.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(2.0),
            endcap = cms.bool(False),
            minTime = cms.double(-20.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(2.0),
            endcap = cms.bool(True),
            minTime = cms.double(-20.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(2.0),
            endcap = cms.bool(False),
            minTime = cms.double(-20.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(2.0),
            endcap = cms.bool(True),
            minTime = cms.double(-20.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(2.),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(False),
            minTime = cms.double(-20.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(2.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(True),
            minTime = cms.double(-20.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(2.),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(False),
            minTime = cms.double(-15.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(2.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(True),
            minTime = cms.double(-15.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(2.),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(False),
            minTime = cms.double(-15.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(2.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(True),
            minTime = cms.double(-15.),
            maxTime = cms.double(25.)
        ),

        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(5.),
            maxEnergy = cms.double(9999999.0),
            endcap = cms.bool(False),
            minTime = cms.double(-5),
            maxTime = cms.double(20.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(5.0),
            maxEnergy = cms.double(9999999.0),
            endcap = cms.bool(True),
            minTime = cms.double(-5),
            maxTime = cms.double(20.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(5.),
            maxEnergy = cms.double(9999999.0),
            endcap = cms.bool(False),
            minTime = cms.double(-5),
            maxTime = cms.double(20.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(5.0),
            maxEnergy = cms.double(9999999.0),
            endcap = cms.bool(True),
            minTime = cms.double(-5),
            maxTime = cms.double(20.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(5.),
            maxEnergy = cms.double(9999999.0),
            endcap = cms.bool(False),
            minTime = cms.double(-5),
            maxTime = cms.double(20.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(5.0),
            maxEnergy = cms.double(9999999.0),
            endcap = cms.bool(True),
            minTime = cms.double(-5),
            maxTime = cms.double(20.)
        )
    )
)

    
