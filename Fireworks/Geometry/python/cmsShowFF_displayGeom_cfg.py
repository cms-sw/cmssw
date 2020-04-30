######################################################################################################
#
# Please run the script with cmsShowFF. E.g.:
#
# cmsShowFF -c $CMSSW_RELEASE_BASE/src/Fireworks/Core/macros/simGeo.fwc cmsShowFF_displayGeom_cfg.py
#
######################################################################################################

import FWCore.ParameterSet.Config as cms


process = cms.Process("FFWDISPLAY")


process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.add_(cms.ESProducer("TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(14)
))


process.source = cms.Source("EmptySource")


