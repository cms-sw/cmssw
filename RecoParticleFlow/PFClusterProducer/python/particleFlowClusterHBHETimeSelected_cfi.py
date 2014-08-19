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
            minTime = cms.double(-25.),
            maxTime = cms.double(30.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(2.0),
            endcap = cms.bool(True),
            minTime = cms.double(-25.),
            maxTime = cms.double(30.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(2.0),
            endcap = cms.bool(False),
            minTime = cms.double(-25.),
            maxTime = cms.double(30.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(2.0),
            endcap = cms.bool(True),
            minTime = cms.double(-25.),
            maxTime = cms.double(30.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(2.0),
            endcap = cms.bool(False),
            minTime = cms.double(-25.),
            maxTime = cms.double(30.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(1.0),
            maxEnergy = cms.double(2.0),
            endcap = cms.bool(True),
            minTime = cms.double(-25.),
            maxTime = cms.double(30.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(2.),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(False),
            minTime = cms.double(-25.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(2.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(True),
            minTime = cms.double(-25.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(2.),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(False),
            minTime = cms.double(-25.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(2.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(True),
            minTime = cms.double(-25.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(2.),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(False),
            minTime = cms.double(-25.),
            maxTime = cms.double(25.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(2.0),
            maxEnergy = cms.double(5.0),
            endcap = cms.bool(True),
            minTime = cms.double(-25.),
            maxTime = cms.double(25.)
        ),


        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(5.),
            maxEnergy = cms.double(10.0),
            endcap = cms.bool(False),
            minTime = cms.double(-22.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(5.0),
            maxEnergy = cms.double(10.0),
            endcap = cms.bool(True),
            minTime = cms.double(-22.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(5.),
            maxEnergy = cms.double(10.0),
            endcap = cms.bool(False),
            minTime = cms.double(-22.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(5.0),
            maxEnergy = cms.double(10.0),
            endcap = cms.bool(True),
            minTime = cms.double(-22.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(5.),
            maxEnergy = cms.double(10.0),
            endcap = cms.bool(False),
            minTime = cms.double(-22.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(5.0),
            maxEnergy = cms.double(10.0),
            endcap = cms.bool(True),
            minTime = cms.double(-22.),
            maxTime = cms.double(15.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(10.),
            maxEnergy = cms.double(9999999.0),
            endcap = cms.bool(False),
            minTime = cms.double(-17.),
            maxTime = cms.double(10.)
        ),
        cms.PSet(
            depth=cms.double(1.0),
            minEnergy = cms.double(10.0),
            maxEnergy = cms.double(99999999.0),
            endcap = cms.bool(True),
            minTime = cms.double(-17.),
            maxTime = cms.double(10.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(10.),
            maxEnergy = cms.double(9999999.0),
            endcap = cms.bool(False),
            minTime = cms.double(-17.),
            maxTime = cms.double(10.)
        ),
        cms.PSet(
            depth=cms.double(2.0),
            minEnergy = cms.double(10.0),
            maxEnergy = cms.double(99999999.0),
            endcap = cms.bool(True),
            minTime = cms.double(-17.),
            maxTime = cms.double(10.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(10.),
            maxEnergy = cms.double(9999999.0),
            endcap = cms.bool(False),
            minTime = cms.double(-17.),
            maxTime = cms.double(10.)
        ),
        cms.PSet(
            depth=cms.double(3.0),
            minEnergy = cms.double(10.0),
            maxEnergy = cms.double(99999999.0),
            endcap = cms.bool(True),
            minTime = cms.double(-17.),
            maxTime = cms.double(10.)
        )

    )
)

    
