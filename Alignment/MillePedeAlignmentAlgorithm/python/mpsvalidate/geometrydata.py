#!/usr/bin/env python

##########################################################################
# Geometry data
##

class GeometryData:
    """ Class which holds the geometry data of a ObjId
    """

    def __init__(self, name, subdetid, discriminator):
        self.name = name
        self.subdetid = subdetid
        self.discriminator = discriminator

# ObjId names from http://cmslxr.fnal.gov/lxr/source/Alignment/CommonAlignment/interface/StructureType.h
data = {-1: GeometryData("notfound", 0, []),
        0: GeometryData("invalid", 0, []),
        1: GeometryData("AlignableDetUnit", 0, []),
        2: GeometryData("AlignableDet", 0, []),
        3: GeometryData("TPBModule", 1, []),
        4: GeometryData("TPBLadder", 1, ["Half", "Layer", "Rod"]),
        5: GeometryData("TPBLayer", 1, ["Half", "Layer"]),
        6: GeometryData("TPBHalfBarrel", 1, ["Half"]),
        7: GeometryData("TPBBarrel", 1, []),
        8: GeometryData("TPEModule", 2, []),
        9: GeometryData("TPEPanel", 2, ["Side", "Half", "Layer", "Blade", "Panel"]),
        10: GeometryData("TPEBlade", 2, ["Side", "Half", "Layer", "Blade"]),
        11: GeometryData("TPEHalfDisk", 2, ["Side", "Half", "Layer"]),
        12: GeometryData("TPEHalfCylinder", 2, ["Side", "Half"]),
        13: GeometryData("TPEEndcap", 2, ["Side"]),
        14: GeometryData("TIBModule", 3, []),
        15: GeometryData("TIBString", 3, []),
        16: GeometryData("TIBSurface", 3, ["Side", "Layer", "Half", "OuterInner"]),
        17: GeometryData("TIBHalfShell", 3, ["Side", "Layer", "Half"]),
        18: GeometryData("TIBLayer", 3, ["Side", "Layer"]),
        19: GeometryData("TIBHalfBarrel", 3, ["Side"]),
        20: GeometryData("TIBBarrel", 3, []),
        21: GeometryData("TIDModule", 4, []),
        22: GeometryData("TIDSide", 4, ["Side", "Layer", "Ring", "OuterInner"]),
        23: GeometryData("TIDRing", 4, ["Side", "Layer", "Ring"]),
        24: GeometryData("TIDDisk", 4, ["Side", "Layer"]),
        25: GeometryData("TIDEndcap", 4, ["Side"]),
        26: GeometryData("TOBModule", 5, []),
        27: GeometryData("TOBRod", 5, ["Side", "Layer", "Rod"]),
        28: GeometryData("TOBLayer", 5, ["Side", "Layer"]),
        29: GeometryData("TOBHalfBarrel", 5, ["Side"]),
        30: GeometryData("TOBBarrel", 5, []),
        31: GeometryData("TECModule", 6, []),
        32: GeometryData("TECRing", 6, ["Side", "Layer", "OuterInner", "Petal", "Ring"]),
        33: GeometryData("TECPetal", 6, ["Side", "Layer", "OuterInner", "Petal"]),
        34: GeometryData("TECSide", 6, ["Side", "Layer", "OuterInner"]),
        35: GeometryData("TECDisk", 6, ["Side", "Layer"]),
        36: GeometryData("TECEndcap", 6, ["Side"]),
        37: GeometryData("Pixel", 0, []),
        38: GeometryData("Strip", 0, []),
        39: GeometryData("Tracker", 0, []),
        100: GeometryData("AlignableDTBarrel", 0, []),
        101: GeometryData("AlignableDTWheel", 0, []),
        102: GeometryData("AlignableDTStation", 0, []),
        103: GeometryData("AlignableDTChamber", 0, []),
        104: GeometryData("AlignableDTSuperLayer", 0, []),
        105: GeometryData("AlignableDTLayer", 0, []),
        106: GeometryData("AlignableCSCEndcap", 0, []),
        107: GeometryData("AlignableCSCStation", 0, []),
        108: GeometryData("AlignableCSCRing", 0, []),
        109: GeometryData("AlignableCSCChamber", 0, []),
        110: GeometryData("AlignableCSCLayer", 0, []),
        111: GeometryData("AlignableMuon", 0, []),
        112: GeometryData("Detector", 0, []),
        1000: GeometryData("Extras", 0, []),
        1001: GeometryData("BeamSpot", 0, [])
        }
