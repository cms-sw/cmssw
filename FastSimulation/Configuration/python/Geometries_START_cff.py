import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.Geometries_cff import *

from FastSimulation.Configuration.FamosSequences_cff import ecalRecHit,hbhereco,horeco,hfreco,famosSimHits

from FastSimulation.CaloRecHitsProducer.CaloRecHits_cff import *
# Apply ECAL/HCAL miscalibration
if(CaloMode==0 or CaloMode==2):
    ecalRecHit.doMiscalib = True
if(CaloMode==0 or CaloMode==1):
    hbhereco.doMiscalib = False
    horeco.doMiscalib = False
    hfreco.doMiscalib = False

# Apply Tracker and Muon misalignment
famosSimHits.ApplyAlignment = True
misalignedTrackerGeometry.applyAlignment = True
misalignedDTGeometry.applyAlignment = True
misalignedCSCGeometry.applyAlignment = True

