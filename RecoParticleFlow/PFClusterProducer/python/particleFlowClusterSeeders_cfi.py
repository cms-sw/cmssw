import FWCore.ParameterSet.Config as cms

localMaxSeeds_EB = cms.PSet(
    algoName = cms.string("LocalMaximum2DSeedFinder"),
    ### seed finding parameters    
    thresholdsByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("ECAL_BARREL"),
                 seedingThreshold = cms.double(0.23),
                 seedingThresholdPt = cms.double(0.0)
                 )
    ),
    nNeighbours = cms.uint32(8)
    )

localMaxSeeds_EE = localMaxSeeds_EB.clone(    
    thresholdsByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("ECAL_ENDCAP"),
                 seedingThreshold = cms.double(0.6),
                 seedingThresholdPt = cms.double(0.15)
                 )
    )
    )

localMaxSeeds_ECAL = localMaxSeeds_EB.clone(
    thresholdsByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("ECAL_ENDCAP"),
                 seedingThreshold = cms.double(0.6),
                 seedingThresholdPt = cms.double(0.15)
                 ),
       cms.PSet( detector = cms.string("ECAL_BARREL"),
                 seedingThreshold = cms.double(0.23),
                 seedingThresholdPt = cms.double(0.0)
                 )
    )
    )

localMaxSeeds_PS = cms.PSet(
    algoName = cms.string("LocalMaximum2DSeedFinder"),
    ### seed finding parameters    
    thresholdsByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("PS1"),
                 seedingThreshold = cms.double(1.2e-4),
                 seedingThresholdPt = cms.double(0.0)
                 ),
       cms.PSet( detector = cms.string("PS2"),
                 seedingThreshold = cms.double(1.2e-4),
                 seedingThresholdPt = cms.double(0.0)
                 )
    ),
    nNeighbours = cms.uint32(4)
    )

localMaxSeeds_HCAL = cms.PSet(
    algoName = cms.string("LocalMaximum2DSeedFinder"),
    thresholdsByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("HCAL_BARREL1"),
                 seedingThreshold = cms.double(0.8),
                 seedingThresholdPt = cms.double(0.0)
                 ),
       cms.PSet( detector = cms.string("HCAL_ENDCAP"),
                 seedingThreshold = cms.double(1.1),
                 seedingThresholdPt = cms.double(0.0)
                 )
       ),
    nNeighbours = cms.uint32(4)
    )

localMaxSeeds_HO = cms.PSet(
    algoName = cms.string("LocalMaximum2DSeedFinder"),
    thresholdsByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("HCAL_BARREL2_RING0"),
                 seedingThreshold = cms.double(1.0),
                 seedingThresholdPt = cms.double(0.0)
                 ),
       cms.PSet( detector = cms.string("HCAL_BARREL2_RING1"),
                 seedingThreshold = cms.double(3.1),
                 seedingThresholdPt = cms.double(0.0)
                 )
       ),
    nNeighbours = cms.uint32(4)
    )

localMaxSeeds_HF = cms.PSet(
    algoName = cms.string("LocalMaximum2DSeedFinder"),
    thresholdsByDetector = cms.VPSet(
       cms.PSet( detector = cms.string("HF_EM"),
                 seedingThreshold = cms.double(1.4),
                 seedingThresholdPt = cms.double(0.0)
                 ),
       cms.PSet( detector = cms.string("HF_HAD"),
                 seedingThreshold = cms.double(1.4),
                 seedingThresholdPt = cms.double(0.0)
                 )
       ),
    nNeighbours = cms.uint32(0)
    )

