import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# Primary vertex smearing.
from IOMC.EventVertexGenerators.VtxSmearedGauss_cfi import *
fastsimPrimaryVertex = 'Realistic8TeV2012' # this is the single place in FastSim where the beamspot scenario is chosen; this choice is propagated to famosSimHits and famosPileUp; in the future, more modularity will be possible when the famosPileUp module will be deprecated
if(fastsimPrimaryVertex=='Realistic8TeV'):
    from FastSimulation.Event.Realistic8TeVCollisionVertexGenerator_cfi import *
elif(fastsimPrimaryVertex=='Realistic8TeV2012'):
    from FastSimulation.Event.Realistic8TeV2012CollisionVertexGenerator_cfi import *
elif(fastsimPrimaryVertex=='Realistic7TeV2011'):
    from FastSimulation.Event.Realistic7TeV2011CollisionVertexGenerator_cfi import *
else: # by default, the currently recommended one
    from FastSimulation.Event.Realistic8TeV2012CollisionVertexGenerator_cfi import *
         
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

