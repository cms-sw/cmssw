import FWCore.ParameterSet.Config as cms

# valid for 13X
# Geometries
#include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cfi" 
# in 150 shall be changed into
# include "Geometry/CMSCommonData/data/cmsIdealGeometryXML.cff"
# from Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi import *
# from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
# Magnetic Field
#include "MagneticField/Engine/data/uniformMagneticField.cfi" 
#include "Geometry/CMSCommonData/data/cmsMagneticFieldXML.cfi"
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
standAloneCosmicMuonsTask = cms.Task(CosmicMuonSeed,standAloneMuons)
standAloneCosmicMuons = cms.Sequence(standAloneCosmicMuonsTask)
standAloneMuons.InputObjects = 'CosmicMuonSeed'
standAloneMuons.STATrajBuilderParameters.NavigationType = 'Direct'
standAloneMuons.TrackLoaderParameters.VertexConstraint = False
# to run only over DT measurements uncomment these cards
CosmicMuonSeed.EnableCSCMeasurement = False
#replace standAloneMuons.STATrajBuilderParameters.RefitterParameters.EnableRPCMeasurement = false
standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableRPCMeasurement = False
#replace standAloneMuons.STATrajBuilderParameters.RefitterParameters.EnableCSCMeasurement = false
standAloneMuons.STATrajBuilderParameters.BWFilterParameters.EnableCSCMeasurement = False


