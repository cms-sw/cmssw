import FWCore.ParameterSet.Config as cms

#
#
#   By Freya Blekman
#
#  default .cfi file 
#
from CondCore.DBCommon.CondDBCommon_cfi import *
#include "CondCore/DBCommon/data/CondDBSetup.cfi"
sipixelcalib_essource = cms.ESSource("PoolDBESSource",
    CondDBCommon,
    #    using CondDBSetup
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelCalibConfigurationRcd'),
        # change the tag appropriately. Look at /afs/cern.ch/cms/Tracker/Pixel/configuration/TestDatabase.txt for existing tag names.
        tag = cms.string('GainCalibration_298')
    ))
)

sipixelcalib_essource.connect = 'oracle://cms_orcoff_int2r/CMS_COND_TIF_PIXELS' ##cms_orcoff_int2r/CMS_COND_TIF_PIXELS"

sipixelcalib_essource.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
#   replace sipixelcalib_essource.connect = "frontier://FrontierDev/CMS_COND_PIXEL"
#   replace sipixelcalib_essource.DBParameters.authenticationMethod = 0
#   seems to have disappeard in 20X
#   replace sipixelcalib_essource.timetype ="runnumber"
sipixelcalib_essource.DBParameters.messageLevel = 0

