import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# Primary vertex smearing.
fastsimPrimaryVertex = 'Realistic8TeV2012'
from IOMC.EventVertexGenerators.VtxSmearedGauss_cfi import *

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


