import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
from Geometry.CMSCommonData.cmsPilot2IdealGeometryXML_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
# Reconstruction geometry services
#  Tracker
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
#  Muon system
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
#  DT
from Geometry.DTGeometry.dtGeometry_cfi import *
#  CSC
from Geometry.CSCGeometry.cscGeometry_cfi import *
#  RPC
from Geometry.RPCGeometry.rpcGeometry_cfi import *
#  Calorimeters
from Geometry.CaloEventSetup.CaloGeometry_cff import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *


