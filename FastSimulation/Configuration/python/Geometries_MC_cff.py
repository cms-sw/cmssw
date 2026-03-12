import FWCore.ParameterSet.Config as cms

def _fastSimGeometryCustomsMC(process):
    # Apply Tracker and Muon misalignment
    process.misalignedTrackerGeometry.applyAlignment = True
    process.misalignedDTGeometry.applyAlignment = True
    process.misalignedCSCGeometry.applyAlignment = True

from Configuration.Eras.Modifier_fastSim_cff import fastSim
modifyGeomMC_fastSim = fastSim.makeProcessModifier(_fastSimGeometryCustomsMC)
