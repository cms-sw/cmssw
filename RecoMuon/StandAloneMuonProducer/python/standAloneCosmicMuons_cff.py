import FWCore.ParameterSet.Config as cms

# valid for 13X
# Geometries
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# in 150 shall be changed into
# include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cff"
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.CSCGeometry.cscGeometry_cfi import *
from Geometry.RPCGeometry.rpcGeometry_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
# Magnetic Field
from MagneticField.Engine.uniformMagneticField_cfi import *
from Geometry.CMSCommonData.cmsMagneticFieldXML_cfi import *
# The services
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
# in 150 shall be used this: 
# include "RecoMuon/TrackingTools/data/MuonTrackLoader.cff"
# Seed generator
from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *
# Standalone muon producer
from RecoMuon.StandAloneMuonProducer.standAloneMuons_cfi import *
# to run only over CSC Measurements uncomment these cards
#replace CosmicMuonSeed.EnableDTMeasurement = false
#replace standAloneMuons.STATrajBuilderParameters.RefitterParameters.EnableRPCMeasurement = false
#replace standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableRPCMeasurement = false
#replace standAloneMuons.STATrajBuilderParameters.RefitterParameters.EnableDTMeasurement = false
#replace standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableDTMeasurement = false
standAloneCosmicMuons = cms.Sequence(CosmicMuonSeed*standAloneMuons)
standAloneMuons.InputObjects = 'CosmicMuonSeed'
standAloneMuons.STATrajBuilderParameters.NavigationType = 'Direct'
standAloneMuons.TrackLoaderParameters.VertexConstraint = False
# to run only over DT measurements uncomment these cards
CosmicMuonSeed.EnableCSCMeasurement = False
standAloneMuons.STATrajBuilderParameters.RefitterParameters.EnableRPCMeasurement = False
standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableRPCMeasurement = False
standAloneMuons.STATrajBuilderParameters.RefitterParameters.EnableCSCMeasurement = False
standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableCSCMeasurement = False

