import FWCore.ParameterSet.Config as cms

from Geometry.CaloTopology.hgcalTopologyTesterEE_cfi import *

hgcalTopologyTesterHEF = hgcalTopologyTesterEE.clone(
    detectorName = "HGCalHESiliconSensitive",
    types        = [0, 0, 1, 1, 1, 1, 2, 2, 2],
    layers       = [1, 2, 9, 3, 4, 5, 6, 7, 8],
    sector1      = [2, 3, 3, 4, 5, 6, 7, 8, 9],
    sector2      = [3, 3, 3, 5, 5, 3, 8, 8, 8],
    cell1        = [0,10,15, 8, 8,10,10,15,15],
    cell2        = [11,7,15,15, 6, 2,15, 8,11] )

hgcalTopologyTesterHEB = hgcalTopologyTesterEE.clone(
    detectorName = "HGCalHEScintillatorSensitive",
    types        = [0,  0,  0, 1, 1, 1, 1,  1,  1],
    layers       = [9, 10, 11,13,14,15,16, 17, 18],
    sector1      = [10,10,  4,14,16, 7, 8,  9, 10],
    cell1        = [1, 10,360, 4,24,40,60,150,288] )
