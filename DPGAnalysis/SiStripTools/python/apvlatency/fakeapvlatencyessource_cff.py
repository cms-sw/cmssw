import FWCore.ParameterSet.Config as cms

from DPGAnalysis.SiStripTools.apvlatency.fakeapvlatencyessource_cfi import *

essapvlatency = cms.ESSource("EmptyESSource",
                              recordName = cms.string("APVLatencyRcd"),
                              firstValid = cms.vuint32(1),
                              iovIsRunNotTime = cms.bool(True)
                              )

