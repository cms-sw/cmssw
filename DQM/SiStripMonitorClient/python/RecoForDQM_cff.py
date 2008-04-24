import FWCore.ParameterSet.Config as cms

# Zero Suppression  ####
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_SimData_cfi import *
# Digitiserr ####
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
# Cluster Finder ####
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_RealData_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# TrackRefitter ####
from RecoTracker.TrackProducer.RefitterWithMaterial_cff import *
# TrackInfo ####
from AnalysisAlgos.TrackInfoProducer.TrackInfoProducer_cfi import *
RecoModulesForSimData = cms.Sequence(siStripZeroSuppression*TrackRefitter*trackinfo)
RecoModulesForTIFData = cms.Sequence(siStripDigis*siStripZeroSuppression*siStripClusters)
siStripDigis.ProductLabel = 'source'
siStripClusters.SiStripQualityLabel = 'test1'
TrackRefitter.TrajectoryInEvent = True
trackinfo.cosmicTracks = 'TrackRefitter'
trackinfo.rechits = 'TrackRefitter'

