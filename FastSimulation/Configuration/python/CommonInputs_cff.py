import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

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

# CaloMode is defined here
# 0: custom local reco bypassing digis, ECAL and HCAL; default before 61x
# 1: as 0, but full digi + std local reco in ECAL 
# 2: as 0, but full digi + std local reco in HCAL
# 3: full digi + std local reco in ECAL and HCAL <---- DEFAULT

CaloMode = 3

# This flag is to switch between GEN-level and SIM/RECO-level pileup mixing

MixingMode = 'GenMixing' # GEN-level <---- DEFAULT
#MixingMode = 'DigiRecoMixing' # SIM/RECO-level; can be used only if CaloMode==3
