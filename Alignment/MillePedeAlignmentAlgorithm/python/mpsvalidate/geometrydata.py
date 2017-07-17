##########################################################################
# Geometry data
##

class GeometryData:
    """ Class which holds the geometry data of a ObjId
    """

    def __init__(self, subdetid = 0, discriminator = ()):
        self.subdetid = subdetid
        self.discriminator = discriminator

# ObjId names from Alignment/CommonAlignment/interface/StructureType.h
data = {-1:   GeometryData(), # notfound
        0:    GeometryData(), # invalid
        1:    GeometryData(), # AlignableDetUnit
        2:    GeometryData(), # AlignableDet
        3:    GeometryData(1),                           # TPBModule
        4:    GeometryData(1, ("Half", "Layer", "Rod")), # TPBLadder
        5:    GeometryData(1, ("Half", "Layer")),        # TPBLayer
        6:    GeometryData(1, ("Half",)),                # TPBHalfBarrel
        7:    GeometryData(1),                           # TPBBarrel
        8:    GeometryData(2),                                              # TPEModule
        9:    GeometryData(2, ("Side", "Half", "Layer", "Blade", "Panel")), # TPEPanel
        10:   GeometryData(2, ("Side", "Half", "Layer", "Blade")),          # TPEBlade
        11:   GeometryData(2, ("Side", "Half", "Layer")),                   # TPEHalfDisk
        12:   GeometryData(2, ("Side", "Half")),                            # TPEHalfCylinder
        13:   GeometryData(2, ("Side",)),                                   # TPEEndcap
        14:   GeometryData(3),                                          # TIBModule
        15:   GeometryData(3),                                          # TIBString
        16:   GeometryData(3, ("Side", "Layer", "Half", "OuterInner")), # TIBSurface
        17:   GeometryData(3, ("Side", "Layer", "Half")),               # TIBHalfShell
        18:   GeometryData(3, ("Side", "Layer")),                       # TIBLayer
        19:   GeometryData(3, ("Side",)),                               # TIBHalfBarrel
        20:   GeometryData(3),                                          # TIBBarrel
        21:   GeometryData(4),                                          # TIDModule
        22:   GeometryData(4, ("Side", "Layer", "Ring", "OuterInner")), # TIDSide
        23:   GeometryData(4, ("Side", "Layer", "Ring")),               # TIDRing
        24:   GeometryData(4, ("Side", "Layer")),                       # TIDDisk
        25:   GeometryData(4, ("Side",)),                               # TIDEndcap
        26:   GeometryData(5),                            # TOBModule
        27:   GeometryData(5, ("Side", "Layer", "Rod")),  # TOBRod
        28:   GeometryData(5, ("Side", "Layer")),         # TOBLayer
        29:   GeometryData(5, ("Side",)),                 # TOBHalfBarrel
        30:   GeometryData(5),                            # TOBBarrel
        31:   GeometryData(6),                                                   # TECModule
        32:   GeometryData(6, ("Side", "Layer", "OuterInner", "Petal", "Ring")), # TECRing
        33:   GeometryData(6, ("Side", "Layer", "OuterInner", "Petal")),         # TECPetal
        34:   GeometryData(6, ("Side", "Layer", "OuterInner")),                  # TECSide
        35:   GeometryData(6, ("Side", "Layer")),                                # TECDisk
        36:   GeometryData(6, ("Side",)),                                        # TECEndcap
        37:   GeometryData(), # Pixel
        38:   GeometryData(), # Strip
        39:   GeometryData(), # Tracker
        100:  GeometryData(), # AlignableDTBarrel
        101:  GeometryData(), # AlignableDTWheel
        102:  GeometryData(), # AlignableDTStation
        103:  GeometryData(), # AlignableDTChamber
        104:  GeometryData(), # AlignableDTSuperLayer
        105:  GeometryData(), # AlignableDTLayer
        106:  GeometryData(), # AlignableCSCEndcap
        107:  GeometryData(), # AlignableCSCStation
        108:  GeometryData(), # AlignableCSCRing
        109:  GeometryData(), # AlignableCSCChamber
        110:  GeometryData(), # AlignableCSCLayer
        111:  GeometryData(), # AlignableMuon
        112:  GeometryData(), # Detector
        1000: GeometryData(), # Extras
        1001: GeometryData(), # BeamSpot
        }
