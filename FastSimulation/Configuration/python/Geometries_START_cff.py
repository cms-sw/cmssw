import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.Geometries_cff import *

from FastSimulation.Configuration.FamosSequences_cff import ecalRecHit,hbhereco,horeco,hfreco,famosSimHits

# Apply ECAL/HCAL miscalibration
ecalRecHit.doMiscalib = True
hbhereco.doMiscalib = True
horeco.doMiscalib = True
hfreco.doMiscalib = True

# Apply Tracker and Muon misalignment
famosSimHits.ApplyAlignment = True
misalignedTrackerGeometry.applyAlignment = True
misalignedDTGeometry.applyAlignment = True
misalignedCSCGeometry.applyAlignment = True

