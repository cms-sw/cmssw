import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.Geometries_cff import *

import FastSimulation.SimplifiedGeometryPropagator.fastSimProducer_cff

# Apply Tracker and Muon misalignment
process.fastSimProducer.detectorDefinition.trackerAlignmentLabel = cms.untracked.string("")
misalignedTrackerGeometry.applyAlignment = True
misalignedDTGeometry.applyAlignment = True
misalignedCSCGeometry.applyAlignment = True

