import FWCore.ParameterSet.Config as cms

from Geometry.CaloEventSetup.CaloGeometryDBReader_cfi import *
#
EcalBarrelGeometryFromDBEP.applyAlignment = True
EcalEndcapGeometryFromDBEP.applyAlignment = True
EcalPreshowerGeometryFromDBEP.applyAlignment = True

HcalGeometryFromDBEP.applyAlignment = True
#CaloTowerGeometryFromDBEP.applyAlignment = True
#ZdcGeometryFromDBEP.applyAlignment = True
#CastorGeometryFromDBEP.applyAlignment = True
