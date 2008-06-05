import FWCore.ParameterSet.Config as cms

# Digitiser ####
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
# Zero Suppression  ####
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
# Cluster Finder ####
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# TrackRefitter ####
from RecoTracker.TrackProducer.RefitterWithMaterial_cff import *
RecoModulesForSimData = cms.Sequence(siStripDigis*siStripZeroSuppression*siStripClusters)
TrackRefitter.TrajectoryInEvent = True

