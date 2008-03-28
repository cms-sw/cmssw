import FWCore.ParameterSet.Config as cms

# define the default tools for the MeasurementTracker by default
# initialize magnetic field #########################
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# initialize geometry #####################
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
# stripCPE #####################
from RecoLocalTracker.SiStripRecHitConverter.StripCPE_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from CalibTracker.SiStripLorentzAngle.SiStripLAFakeESSource_cfi import *
# pixelCPE #####################
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
# region cabling ######################
from CondCore.DBCommon.CondDBSetup_cfi import *
from EventFilter.SiStripRawToDigi.SiStripFrontierCabling_cfi import *
from CalibTracker.SiStripConnectivity.SiStripConnectivity_cfi import *
from CalibTracker.SiStripConnectivity.SiStripRegionConnectivity_cfi import *
from RecoTracker.MeasurementDet.OnDemandMeasurementTrackerESProducer_cfi import *

