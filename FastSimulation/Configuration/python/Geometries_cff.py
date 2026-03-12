import FWCore.ParameterSet.Config as cms
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import trackerGeometry as _trackerGeometry
from Geometry.DTGeometryBuilder.dtGeometry_cfi import DTGeometryESModule as _DTGeometryESModule
from Geometry.CSCGeometryBuilder.cscGeometry_cfi import CSCGeometryESModule as _CSCGeometryESModule

def _fastSimGeometryCustoms(process):
    # The tracker geometry left-over (for aligned/misaligned geometry)
    # The goemetry used for reconstruction must not be misaligned.
    process.trackerGeometry.applyAlignment = False
    # Create a misaligned geometry for simulation
    process.misalignedTrackerGeometry = _trackerGeometry.clone()
    # The misalignment is not applied by default
    process.misalignedTrackerGeometry.applyAlignment = False
    # Label of the produced TrackerGeometry:
    process.misalignedTrackerGeometry.appendToDataLabel = 'MisAligned'

    # The DT geometry left-over (for aligned/misaligned geometry)
    # The geometry used for reconstruction must not be misaligned.
    process.DTGeometryESModule.applyAlignment = False
    # Create a misaligned geometry for simulation
    process.misalignedDTGeometry = _DTGeometryESModule.clone()
    # The misalignment is not applied by default
    process.misalignedDTGeometry.applyAlignment = False
    # Label of the produced DTGeometry:
    process.misalignedDTGeometry.appendToDataLabel = 'MisAligned'

    # The CSC geometry left-over (for aligned/misaligned geometry)
    # The geometry used for reconstruction must not be misaligned.
    process.CSCGeometryESModule.applyAlignment = False
    # Create a misaligned geometry for simulation
    process.misalignedCSCGeometry = _CSCGeometryESModule.clone()
    # The misalignment is not applied by default
    process.misalignedCSCGeometry.applyAlignment = False
    # Label of the produced CSCGeometry:
    process.misalignedCSCGeometry.appendToDataLabel = 'MisAligned'

    # Reconstruction and Interaction tracker geometries
    process.load("FastSimulation.Configuration.TrackerRecoGeometryESProducer_cfi")

    # The Calo geometry service model left-over
    process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

    # The muon geometry left-over
    process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")

from Configuration.Eras.Modifier_fastSim_cff import fastSim
modifyGeom_fastSim = fastSim.makeProcessModifier(_fastSimGeometryCustoms)
