import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# Tracking Monitor 
#-------------------------------------------------
from DQM.TrackingMonitor.TrackingMonitor_cfi import *

# properties
TrackMon.OutputMEsInRootFile 	= cms.bool(True)
TrackMon.OutputFileName         = cms.string('TrackingMonitorSeedMultiplicity.root')
TrackMon.MeasurementState       = cms.string('ImpactPoint')
TrackMon.FolderName    = cms.string('Tracking/GenTk')
TrackMon.AlgoName      = cms.string('Seed')

# out of the box
# ---------------------------------------------------------------------------#


TrackMonStep0 = TrackMon.clone()
TrackMonStep0.SeedProducer  = cms.InputTag("newSeedFromTriplets")
TrackMonStep0.doSeedNumberHisto = cms.bool(True)
TrackMonStep0.doSeedVsClusterHisto = cms.bool(True)
TrackMonStep0.doSeedPTHisto  = cms.bool(True)
TrackMonStep0.TkSeedSizeBin = cms.int32(200)
TrackMonStep0.TkSeedSizeMax = cms.double(1500)                         
TrackMonStep0.TkSeedSizeMin = cms.double(0)

TrackMonStep1 = TrackMon.clone()
TrackMonStep1.SeedProducer  = cms.InputTag("newSeedFromPairs")
TrackMonStep1.doSeedNumberHisto = cms.bool(True)
TrackMonStep1.doSeedVsClusterHisto = cms.bool(True)
TrackMonStep1.doSeedPTHisto  = cms.bool(True)
TrackMonStep1.TkSeedSizeBin = cms.int32(200)
TrackMonStep1.TkSeedSizeMax = cms.double(100.e3)                         
TrackMonStep1.TkSeedSizeMin = cms.double(0)


TrackMonStep2 = TrackMon.clone()
TrackMonStep2.SeedProducer  = cms.InputTag("secTriplets")
TrackMonStep2.doSeedNumberHisto = cms.bool(True)
TrackMonStep2.doSeedVsClusterHisto = cms.bool(True)
TrackMonStep2.doSeedPTHisto  = cms.bool(True)
TrackMonStep2.TkSeedSizeBin = cms.int32(200)
TrackMonStep2.TkSeedSizeMax = cms.double(30.e3)                         
TrackMonStep2.TkSeedSizeMin = cms.double(0)

TrackMonStep3 = TrackMon.clone()
TrackMonStep3.SeedProducer  = cms.InputTag("thTriplets")
TrackMonStep3.doSeedNumberHisto = cms.bool(True)
TrackMonStep3.doSeedVsClusterHisto = cms.bool(True)
TrackMonStep3.doSeedPTHisto  = cms.bool(True)
TrackMonStep3.TkSeedSizeBin = cms.int32(200)
TrackMonStep3.TkSeedSizeMax = cms.double(100.e3)                         
TrackMonStep3.TkSeedSizeMin = cms.double(0)

TrackMonStep4 = TrackMon.clone()
TrackMonStep4.SeedProducer  = cms.InputTag("fourthPLSeeds")
TrackMonStep4.doSeedNumberHisto = cms.bool(True)
TrackMonStep4.doSeedVsClusterHisto = cms.bool(True)
TrackMonStep4.doSeedPTHisto  = cms.bool(True)
TrackMonStep4.TkSeedSizeBin = cms.int32(200)
TrackMonStep4.TkSeedSizeMax = cms.double(100.e3)                         
TrackMonStep4.TkSeedSizeMin = cms.double(0)


TrackMonStep5 = TrackMon.clone()
TrackMonStep5.SeedProducer  = cms.InputTag("fifthSeeds")
TrackMonStep5.doSeedNumberHisto = cms.bool(True)
TrackMonStep5.doSeedVsClusterHisto = cms.bool(True)
TrackMonStep5.doSeedPTHisto  = cms.bool(True)
TrackMonStep5.TkSeedSizeBin = cms.int32(200)
TrackMonStep5.TkSeedSizeMax = cms.double(30.e3)                         
TrackMonStep5.TkSeedSizeMin = cms.double(0)

# out of the box
trkmonootb = cms.Sequence(
      TrackMonStep0
    * TrackMonStep1
    * TrackMonStep2
    * TrackMonStep4
    * TrackMonStep5 
)



# all paths
trkmon = cms.Sequence(
      trkmonootb
)

