import FWCore.ParameterSet.Config as cms

# Magnetic Field
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# Geometries
from Geometry.CMSCommonData.cmsIdealGeometryXML_cff import *
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.CSCGeometry.cscGeometry_cfi import *
from Geometry.RPCGeometry.rpcGeometry_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
# Stand Alone Muons Producer
from RecoMuon.StandAloneMuonProducer.standAloneMuons_cfi import *

