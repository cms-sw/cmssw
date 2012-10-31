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
#from FastSimulation.Configuration.Geometries_cff import *

#The Magnetic Field ESProducer's
from FastSimulation.ParticlePropagator.MagneticFieldMapESProducer_cfi import *

# The muon tracker trajectory, to be fit without rechit refit
from RecoMuon.GlobalTrackingTools.GlobalTrajectoryBuilderCommon_cff import GlobalTrajectoryBuilderCommon
GlobalTrajectoryBuilderCommon.TrackerRecHitBuilder = 'WithoutRefit'
GlobalTrajectoryBuilderCommon.TrackTransformer.TrackerRecHitBuilder = 'WithoutRefit'

# ECAL severity
from RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi import *

# CaloMode is defined in FastSimulation.CaloRecHitsProducer.CaloRecHits_cff
# 0: custom local reco bypassing digis, ECAL and HCAL
# 1: as 0, but full digi + std local reco in ECAL
# 2: as 0, but full digi + std local reco in HCAL
# 3: full digi + std local reco in ECAL and HCAL
from FastSimulation.CaloRecHitsProducer.CaloRecHits_cff import *
from FastSimulation.Calorimetry.Calorimetry_cff import *
if(CaloMode==0):
    FamosCalorimetryBlock.Calorimetry.ECAL.Digitizer = False
    FamosCalorimetryBlock.Calorimetry.HCAL.Digitizer = False
if(CaloMode==1):
    FamosCalorimetryBlock.Calorimetry.ECAL.Digitizer = True
    FamosCalorimetryBlock.Calorimetry.HCAL.Digitizer = False
if(CaloMode==2):
    FamosCalorimetryBlock.Calorimetry.ECAL.Digitizer = False
    FamosCalorimetryBlock.Calorimetry.HCAL.Digitizer = True
    FamosCalorimetryBlock.Calorimetry.HCAL.smearTimeHF = True
if(CaloMode==3):
    FamosCalorimetryBlock.Calorimetry.ECAL.Digitizer = True
    FamosCalorimetryBlock.Calorimetry.HCAL.Digitizer = True
    FamosCalorimetryBlock.Calorimetry.HCAL.smearTimeHF = True
