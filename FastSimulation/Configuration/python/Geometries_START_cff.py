import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.Geometries_cff import *

import FastSimulation.EventProducer.FamosSimHits_cff

# Apply Tracker and Muon misalignment
FastSimulation.EventProducer.FamosSimHits_cff.famosSimHits.ApplyAlignment = True
misalignedTrackerGeometry.applyAlignment = True
misalignedDTGeometry.applyAlignment = True
misalignedCSCGeometry.applyAlignment = True

