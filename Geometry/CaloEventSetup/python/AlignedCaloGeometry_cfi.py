import FWCore.ParameterSet.Config as cms

from Geometry.CaloEventSetup.CaloGeometry_cff import *
#
EcalBarrelGeometryEP.applyAlignment = True
EcalEndcapGeometryEP.applyAlignment = True
EcalPreshowerGeometryEP.applyAlignment = True
