import FWCore.ParameterSet.Config as cms

# Geometries
from Geometry.CMSCommonData.cmsIdealGeometryXML_cff import *
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.CSCGeometry.cscGeometry_cfi import *
from Geometry.RPCGeometry.rpcGeometry_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
# The services
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *

