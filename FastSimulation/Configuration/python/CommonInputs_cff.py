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

# The condDB setup (the global tag refers to DevDB, IntDB or ProDB whenever needed)
#from Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi import *

# not needed any longer in 30X
#from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *

#from CalibMuon.CSCCalibration.CSC_BadChambers_cfi import *
#hcal_db_producer = cms.ESProducer("HcalDbProducer",
#    dump = cms.untracked.vstring(''),
#    file = cms.untracked.string('')
#)

#es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
#    toGet = cms.untracked.vstring('GainWidths')
#)

from RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi import *


