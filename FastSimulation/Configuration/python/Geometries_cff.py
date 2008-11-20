import FWCore.ParameterSet.Config as cms

# The CMS Geometry files
# Pilot 2 geometry doesn't contain the preshower
#from Configuration.StandardSequences.GeometryPilot2_cff import *

# To use the "full" CMS geometry, comment the prevous line, and uncomment the following one:
from Configuration.StandardSequences.Geometry_cff import *

# The tracker geometry left-over (for aligned/misaligned geometry)
# The goemetry used for reconstruction must not be misaligned.
TrackerDigiGeometryESModule.applyAlignment = False
# Create a misaligned geometry for simulation
misalignedTrackerGeometry = Geometry.TrackerGeometryBuilder.trackerGeometry_cfi.TrackerDigiGeometryESModule.clone()
# The misalignment is not applied by default
misalignedTrackerGeometry.applyAlignment = False
# Label of the produced TrackerGeometry:
misalignedTrackerGeometry.appendToDataLabel = 'MisAligned'

# Reconstruction and Interaction tracker geometries
from FastSimulation.Configuration.TrackerRecoGeometryESProducer_cfi import *
from FastSimulation.TrackerSetup.TrackerInteractionGeometryESProducer_cfi import *

# The Calo geometry service model left-over
from Geometry.CaloEventSetup.CaloTopology_cfi import *

# The muon geometry left-over
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
