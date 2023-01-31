import FWCore.ParameterSet.Config as cms

from Geometry.CaloTopology.hgcalTBTopologyTesterEE_cfi import *

hgcalTBTopologyTesterHEF = hgcalTBTopologyTesterEE.clone(
    detectorName = "HGCalHESiliconSensitive",
    types        = [0, 0, 1, 1, 1, 1, 2, 2, 2],
    layers       = [1, 2, 9, 3, 4, 5, 6, 7, 8],
    sector       = [2, 3, 3, 4, 5, 6, 7, 8, 9],
    cells        = [0,10,15, 8, 8,10,10,15,15] )
