import FWCore.ParameterSet.Config as cms

# The CMS Geometry files
# Pilot 2 geometry doesn't contain the preshower
#from Configuration.StandardSequences.GeometryPilot2_cff import *

# To use the "full" CMS geometry, comment the prevous line, and uncomment the following one:
#####from Configuration.StandardSequences.Geometry_cff import *
from Configuration.StandardSequences.GeometryDB_cff import *


# The tracker geometry left-over (for aligned/misaligned geometry)
# The goemetry used for reconstruction must not be misaligned.
trackerGeometryDB.applyAlignment = False
# Create a misaligned geometry for simulation
misalignedTrackerGeometry = Geometry.TrackerGeometryBuilder.trackerGeometryDB_cfi.trackerGeometryDB.clone()
# The misalignment is not applied by default
misalignedTrackerGeometry.applyAlignment = False
# Label of the produced TrackerGeometry:
misalignedTrackerGeometry.appendToDataLabel = 'MisAligned'

# The DT geometry left-over (for aligned/misaligned geometry)
# The geometry used for reconstruction must not be misaligned.
DTGeometryESModule.applyAlignment = False
# Create a misaligned geometry for simulation
misalignedDTGeometry = Geometry.DTGeometryBuilder.dtGeometryDB_cfi.DTGeometryESModule.clone()
# The misalignment is not applied by default
misalignedDTGeometry.applyAlignment = False
# Label of the produced DTGeometry:
misalignedDTGeometry.appendToDataLabel = 'MisAligned'

# The CSC geometry left-over (for aligned/misaligned geometry)
# The geometry used for reconstruction must not be misaligned.
CSCGeometryESModule.applyAlignment = False
# Create a misaligned geometry for simulation
misalignedCSCGeometry = Geometry.CSCGeometryBuilder.cscGeometryDB_cfi.CSCGeometryESModule.clone()
# The misalignment is not applied by default
misalignedCSCGeometry.applyAlignment = False
# Label of the produced CSCGeometry:
misalignedCSCGeometry.appendToDataLabel = 'MisAligned'



# Reconstruction and Interaction tracker geometries
from FastSimulation.Configuration.TrackerRecoGeometryESProducer_cfi import *
from FastSimulation.TrackerSetup.TrackerInteractionGeometryESProducer_cfi import *

# The Calo geometry service model left-over
from Geometry.CaloEventSetup.CaloTopology_cfi import *

# The muon geometry left-over
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
