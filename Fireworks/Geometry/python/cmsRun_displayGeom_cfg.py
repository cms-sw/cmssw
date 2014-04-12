#################################################
#
# Please run the script with cmsRun:
# 
# cmsRun displayGeom_cfg.py
#
#################################################

import FWCore.ParameterSet.Config as cms

process = cms.Process("DISPLAY")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")



process.source = cms.Source("EmptySource")

process.EveService = cms.Service("EveService")

### Extractor of geometry needed to display it in Eve.
### Required for "DummyEvelyser".
process.add_( cms.ESProducer(
        "TGeoMgrFromDdd",
        verbose = cms.untracked.bool(False),
        level   = cms.untracked.int32(8)
))

process.dump = cms.EDAnalyzer(
    "DisplayGeom"

)

process.p = cms.Path(process.dump)
