import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.Geometries_cff import *

from FastSimulation.Configuration.FamosSequences_cff import famosSimHits

# Apply Tracker and Muon misalignment
famosSimHits.ApplyAlignment = True
misalignedTrackerGeometry.applyAlignment = True
misalignedDTGeometry.applyAlignment = True
misalignedCSCGeometry.applyAlignment = True
