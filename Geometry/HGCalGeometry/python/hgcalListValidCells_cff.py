import FWCore.ParameterSet.Config as cms

from Geometry.HGCalGeometry.hgcalListValidCellsEE_cfi import *

hgcalListValidCellsHE = hgcalListValidCellsEE.clone(
    detector = "HGCalHESiliconSensitive",
)
