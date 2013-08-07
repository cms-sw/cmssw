import FWCore.ParameterSet.Config as cms

# CMS Sim and Reco Geometry for 2017 scenario
from Configuration.Geometry.GeometryExtended2017Reco_cff import *

# The geometry used for reconstruction must not be misaligned.
trackerSLHCGeometry.applyAlignment = cms.bool(False)
# Create a misaligned geometry for simulation
# Try to change the name of the file.
misalignedTrackerGeometry = Geometry.TrackerGeometryBuilder.trackerSLHCGeometry_cfi.trackerSLHCGeometry.clone()
# The misalignment is not applied by default
misalignedTrackerGeometry.applyAlignment = cms.bool(False)
# Label of the produced TrackerGeometry:
misalignedTrackerGeometry.appendToDataLabel = 'MisAligned'

# The DT geometry left-over (for aligned/misaligned geometry)
# The geometry used for reconstruction must not be misaligned.
DTGeometryESModule.applyAlignment = False
# Create a misaligned geometry for simulation
misalignedDTGeometry = Geometry.DTGeometryBuilder.dtGeometry_cfi.DTGeometryESModule.clone()
# The misalignment is not applied by default
misalignedDTGeometry.applyAlignment = False
# Label of the produced DTGeometry:
misalignedDTGeometry.appendToDataLabel = 'MisAligned'

# The CSC geometry left-over (for aligned/misaligned geometry)
# The geometry used for reconstruction must not be misaligned.
CSCGeometryESModule.applyAlignment = False
# Create a misaligned geometry for simulation
misalignedCSCGeometry = Geometry.CSCGeometryBuilder.cscGeometry_cfi.CSCGeometryESModule.clone()
# The misalignment is not applied by default
misalignedCSCGeometry.applyAlignment = False
# Label of the produced CSCGeometry:
misalignedCSCGeometry.appendToDataLabel = 'MisAligned'

# Reconstruction and Interaction tracker geometries
from FastSimulation.Configuration.TrackerRecoGeometryESProducer_cfi import *
from FastSimulation.TrackerSetup.TrackerInteractionGeometryESProducer_cfi import *
