import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBSetup_cfi import *
# the following is needed for non PoolDBESSources (fake calibrations)
#
from CalibTracker.SiStripESProducers.SiStripPedestalsFakeSource_cfi import *
from CalibTracker.SiStripESProducers.SiStripQualityFakeESSource_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.CSCGeometry.cscGeometry_cfi import *
from CalibCalorimetry.EcalLaserCorrection.ecalLaserCorrectionService_cfi import *
from CalibTracker.SiStripESProducers.SiStripGainESProducer_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('GainWidths', 
        'channelQuality', 
        'ZSThresholds')
)

sistripconn = cms.ESProducer("SiStripConnectivity")

# end fake calibrations
GlobalTag = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    connect = cms.string('frontier://FrontierInt/CMS_COND_20X_GLOBALTAG'), ##FrontierInt/CMS_COND_20X_GLOBALTAG"

    globaltag = cms.untracked.string('IDEAL::All'),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
)

TrackerDigiGeometryESModule.applyAlignment = True
DTGeometryESModule.applyAlignment = True
CSCGeometryESModule.applyAlignment = True

