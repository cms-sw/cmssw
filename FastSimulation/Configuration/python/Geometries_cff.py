import FWCore.ParameterSet.Config as cms

# The CMS Geometry files
from Configuration.StandardSequences.Geometry_cff import *

# The tracker geometry left-over (for misaligned geometry)
misalignedTrackerGeometry = Geometry.TrackerGeometryBuilder.trackerGeometry_cfi.TrackerDigiGeometryESModule.clone()
# The misalignment won't be applied
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
