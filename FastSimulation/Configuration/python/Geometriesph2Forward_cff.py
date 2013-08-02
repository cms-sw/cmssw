import FWCore.ParameterSet.Config as cms


# The CMS Geometry files
# Pilot 2 geometry doesn't contain the preshower
#from Configuration.StandardSequences.GeometryPilot2_cff import *

# To use the "full" CMS geometry, comment the prevous line, and uncomment the following one:
#####from Configuration.StandardSequences.Geometry_cff import *
#from Configuration.StandardSequences.GeometryDB_cff import *
#from Configuration.Geometry.GeometryExtended2017Reco_cff import *
from Configuration.Geometry.GeometryExtendedPhase2TkBERecoForward_cff import *



#TrackerDigiGeometryESModule.applyAlignement = False
# The tracker geometry left-over (for aligned/misaligned geometry)
# The goemetry used for reconstruction must not be misaligned.
trackerSLHCGeometry.applyAlignment = cms.bool(False)
# Create a misaligned geometry for simulation
#misalignedTrackerGeometry = Geometry.TrackerGeometryBuilder.trackerGeometry_cfi.trackerGeometry.clone()
#Try to change the name of the file.
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

# The Calo geometry service model left-over
from Geometry.CaloEventSetup.CaloTopology_cfi import *

# The muon geometry left-over
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
