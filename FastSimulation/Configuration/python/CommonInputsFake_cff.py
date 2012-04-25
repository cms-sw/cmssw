import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# Primary vertex smearing.
fastsimPrimaryVertex = 'Realistic8TeV2012'
from IOMC.EventVertexGenerators.VtxSmearedGauss_cfi import *

# The Geometries
from FastSimulation.Configuration.Geometries_cff import *

#The Magnetic Field ESProducer's 
from FastSimulation.ParticlePropagator.MagneticFieldMapESProducer_cfi import *

# The muon digi calibration
from CalibMuon.Configuration.DT_FakeConditions_cff import *
from CalibMuon.Configuration.CSC_FakeDBConditions_cff import *

# Services from the CondDB
from CondCore.DBCommon.CondDBSetup_cfi import *
from RecoVertex.BeamSpotProducer.BeamSpotFakeConditionsEarly10TeVCollision_cff import *
from RecoBTag.Configuration.RecoBTag_FakeConditions_cff import *
from RecoBTau.Configuration.RecoBTau_FakeConditions_cff import *
from CalibCalorimetry.Configuration.Hcal_FakeConditions_cff import *
from CalibCalorimetry.Configuration.Ecal_FakeConditions_cff import *
from CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff import *
from CalibMuon.Configuration.RPC_FakeConditions_cff import *

# Muon Tracking
from RecoMuon.GlobalTrackingTools.GlobalTrajectoryBuilderCommon_cff import *
GlobalTrajectoryBuilderCommon.TrackerRecHitBuilder = 'WithoutRefit'
GlobalTrajectoryBuilderCommon.TrackTransformer.TrackerRecHitBuilder = 'WithoutRefit'

