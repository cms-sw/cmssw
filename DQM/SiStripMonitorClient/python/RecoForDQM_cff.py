import FWCore.ParameterSet.Config as cms

# Digitiser ####
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
# Local Reco ####    
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
# Track Reconstruction ########
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
BeamSpotEarlyCollision = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
    ),
    timetype = cms.string('runtime'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('BeamSpotObjectsRcd'),
        tag = cms.string('EarlyCollision_5p3cm_mc')
    )),
    connect = cms.string('frontier://Frontier/CMS_COND_20X_BEAMSPOT') ##Frontier/CMS_COND_20X_BEAMSPOT"

)

RecoModulesForTIFData = cms.Sequence(siPixelDigis*siStripDigis*offlineBeamSpot*trackerlocalreco*ctftracksP5)
siStripDigis.ProductLabel = 'source'
siPixelDigis.InputLabel = 'source'
siStripClusters.SiStripQualityLabel = 'test1'

