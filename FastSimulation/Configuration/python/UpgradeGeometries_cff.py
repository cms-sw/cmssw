import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.Geometries_SLHC_cff import *

from FastSimulation.Configuration.FamosSequences_cff import ecalRecHit,hbhereco,horeco,hfreco,famosSimHits

# Apply ECAL/HCAL miscalibration
ecalRecHit.doMiscalib = True
hbhereco.doMiscalib = False
horeco.doMiscalib = False
hfreco.doMiscalib = False

# Apply Tracker and Muon misalignment
famosSimHits.ApplyAlignment = True
misalignedTrackerGeometry.applyAlignment = False
misalignedDTGeometry.applyAlignment = True
misalignedCSCGeometry.applyAlignment = True

