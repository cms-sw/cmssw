import FWCore.ParameterSet.Config as cms

from Geometry.EcalAlgo.EcalEndcapGeometry_cfi import *
from Geometry.EcalAlgo.EcalPreshowerGeometry_cfi import *
from Geometry.EcalAlgo.EcalBarrelGeometry_cfi import *
#
# This cfi should be included to build the ALIGNED ECAL + HCAL geometry model
#
EcalBarrelGeometryEP.applyAlignment = True
EcalEndcapGeometryEP.applyAlignment = True
EcalPreshowerGeometryEP.applyAlignment = True
