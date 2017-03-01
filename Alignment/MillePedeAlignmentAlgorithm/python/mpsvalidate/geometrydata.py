#!/usr/bin/env python

##########################################################################
# Geometry data
##

class GeometryData:
    """ Class which holds the geometry data of a ObjId
    """

    def __init__(self, name, subdetid, discriminator, ndiscriminator):
        self.name = name
        self.subdetid = subdetid
        self.discriminator = discriminator
        self.ndiscriminator = ndiscriminator

# ObjId names from http://cmslxr.fnal.gov/lxr/source/Alignment/CommonAlignment/interface/StructureType.h
data = {-1: GeometryData("notfound", 0, [], []),
        0: GeometryData("invalid", 0, [], []),
        1: GeometryData("AlignableDetUnit", 0, [], []),
        2: GeometryData("AlignableDet", 0, [], []),
        3: GeometryData("TPBModule", 1, [], []),
        4: GeometryData("TPBLadder", 1, ["half", "layer", "rod"], [2, 3, 44]),
        5: GeometryData("TPBLayer", 1, ["half", "layer"], [2, 3]),
        6: GeometryData("TPBHalfBarrel", 1, ["half"], [2]),
        7: GeometryData("TPBBarrel", 1, [], []),
        8: GeometryData("TPEModule", 2, [], []),
        9: GeometryData("TPEPanel", 2, ["side", "half", "layer", "blade", "panel"], [2, 2, 2, 24, 2]),
        10: GeometryData("TPEBlade", 2, ["side", "half", "layer", "blade"], [2, 2, 2, 24]),
        11: GeometryData("TPEHalfDisk", 2, ["side", "half", "layer"], [2, 2, 2]),
        12: GeometryData("TPEHalfCylind", 2, ["side", "half"], [2, 2]),
        13: GeometryData("TPEEndcap", 2, ["side"], [2]),
        14: GeometryData("TIBModule", 3, [], []),
        15: GeometryData("TIBString", 3, [], []),
        16: GeometryData("TIBSurface", 3, ["side", "layer", "half", "outerinner"], [2, 4, 2, 2]),
        17: GeometryData("TIBHalfShell", 3, ["side", "layer", "half"], [2, 4, 2]),
        18: GeometryData("TIBLayer", 3, ["side", "layer"], [2, 4]),
        19: GeometryData("TIBHalfBarrel", 3, ["side"], [2]),
        20: GeometryData("TIBBarrel", 3, [], []),
        21: GeometryData("TIDModule", 4, [], []),
        22: GeometryData("TIDSide", 4, ["side", "layer", "ring", "outerinner"], [2, 3, 3, 2]),
        23: GeometryData("TIDRing", 4, ["side", "layer", "ring"], [2, 3, 3]),
        24: GeometryData("TIDDisk", 4, ["side", "layer"], [2, 3]),
        25: GeometryData("TIDEndcap", 4, ["side"], [2]),
        26: GeometryData("TOBModule", 5, [], []),
        27: GeometryData("TOBRod", 5, ["side", "layer", "rod"], [2, 6, 74]),
        28: GeometryData("TOBLayer", 5, ["side", "layer"], [2, 6]),
        29: GeometryData("TOBHalfBarrel", 5, ["side"], [2]),
        30: GeometryData("TOBBarrel", 5, [], []),
        31: GeometryData("TECModule", 6, [], []),
        32: GeometryData("TECRing", 6, ["side", "layer", "outerinner", "petal", "ring"], [2, 9, 2, 7]),
        33: GeometryData("TECPetal", 6, ["side", "layer", "outerinner", "petal"], [2, 9, 2, 8]),
        34: GeometryData("TECSide", 6, ["side", "layer", "outerinner"], [2, 9, 2]),
        35: GeometryData("TECDisk", 6, ["side", "layer"], [2, 9]),
        36: GeometryData("TECEndcap", 6, ["side"], [2]),
        37: GeometryData("Pixel", 0, [], []),
        38: GeometryData("Strip", 0, [], []),
        39: GeometryData("Tracker", 0, [], []),
        100: GeometryData("AlignableDTBarrel", 0, [], []),
        101: GeometryData("AlignableDTWheel", 0, [], []),
        102: GeometryData("AlignableDTStation", 0, [], []),
        103: GeometryData("AlignableDTChamber", 0, [], []),
        104: GeometryData("AlignableDTSuperLayer", 0, [], []),
        105: GeometryData("AlignableDTLayer", 0, [], []),
        106: GeometryData("AlignableCSCEndcap", 0, [], []),
        107: GeometryData("AlignableCSCStation", 0, [], []),
        108: GeometryData("AlignableCSCRing", 0, [], []),
        109: GeometryData("AlignableCSCChamber", 0, [], []),
        110: GeometryData("AlignableCSCLayer", 0, [], []),
        111: GeometryData("AlignableMuon", 0, [], []),
        112: GeometryData("Detector", 0, [], []),
        1000: GeometryData("Extras", 0, [], []),
        1001: GeometryData("BeamSpot", 0, [], [])
        }
