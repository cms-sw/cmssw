import FWCore.ParameterSet.Config as cms

TrackerShortTermPresigmas = cms.PSet(
    Presigmas = cms.VPSet(cms.PSet(
        presigma = cms.double(0.0013),
        Selector = cms.PSet(
            alignParams = cms.vstring('PixelHalfBarrelDets,111000')
        )
    ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrelDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0005),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrelLadders,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrelLadders,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.001),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrelLayers,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrelLayers,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrelLayers,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0005),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.00025),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECPetals,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECPetals,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0005),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECLayers,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(5e-06),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECLayers,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECLayers,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.02),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.02),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBRods,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBRods,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0105),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBLayers,110000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBLayers,001000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(9e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBLayers,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBLayers,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0105),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.03),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDRings,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDRings,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.04),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDLayers,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0001),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDLayers,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDLayers,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.01),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.01),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBRods,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBRods,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0067),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBLayers,110000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBLayers,001000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(5.9e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBLayers,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBLayers,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.005),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.01),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECPetals,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECPetals,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0057),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECLayers,110000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECLayers,001000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(4.6e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECLayers,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECLayers,000110')
            )
        ))# end of VPSet Presigmas
)# end TrackerShortTermPresigmas
#
#
#
#
TrackerORCAShortTermPresigmas = cms.PSet(
    Presigmas = cms.VPSet(cms.PSet(
        presigma = cms.double(0.0013),
        Selector = cms.PSet(
            alignParams = cms.vstring('PixelHalfBarrelDets,111000')
        )
    ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrelDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0005),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrelLadders,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrelLadders,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.001),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrels,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrels,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrels,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0005),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.00025),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECPetals,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECPetals,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0005),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXEndCaps,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(5e-06),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXEndCaps,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXEndCaps,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.02),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.02),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBRods,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBRods,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0105),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBHalfBarrels,110000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBHalfBarrels,001000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(9e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBHalfBarrels,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBHalfBarrels,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0105),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.03),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDRings,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDRings,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.04),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDs,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0001),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDs,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDs,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.01),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.01),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBRods,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBRods,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0067),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBHalfBarrels,110000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBHalfBarrels,001000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(5.9e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBHalfBarrels,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBHalfBarrels,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.005),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.01),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECPetals,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECPetals,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0057),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECs,110000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECs,001000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(4.6e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECs,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECs,000110')
            )
        ))# end of VPSet Presigmas
) # end TrackerORCAShortTermPresigmas
#
#
#
#
TrackerORCAShortTermPresigmasDetBy10 = cms.PSet(
    Presigmas = cms.VPSet(cms.PSet(
        presigma = cms.double(0.00013),
        Selector = cms.PSet(
            alignParams = cms.vstring('PixelHalfBarrelDets,111000')
        )
    ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrelDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0005),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrelLadders,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrelLadders,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.001),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrels,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrels,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PixelHalfBarrels,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(5e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.00025),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECPetals,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXECPetals,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0005),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXEndCaps,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(5e-06),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXEndCaps,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('PXEndCaps,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.002),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.02),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBRods,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBRods,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0105),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBHalfBarrels,110000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBHalfBarrels,001000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(9e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBHalfBarrels,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIBHalfBarrels,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.00105),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.03),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDRings,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDRings,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.04),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDs,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0001),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDs,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TIDs,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.001),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.01),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBRods,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBRods,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0067),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBHalfBarrels,110000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBHalfBarrels,001000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(5.9e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBHalfBarrels,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TOBHalfBarrels,000110')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0005),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECDets,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECDets,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.01),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECPetals,111000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECPetals,000111')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.0057),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECs,110000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(0.05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECs,001000')
            )
        ), 
        cms.PSet(
            presigma = cms.double(4.6e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECs,000001')
            )
        ), 
        cms.PSet(
            presigma = cms.double(1e-05),
            Selector = cms.PSet(
                alignParams = cms.vstring('TECs,000110')
            )
        ))# end of VPSet Presigmas
) # end TrackerORCAShortTermPresigmasDetBy10


