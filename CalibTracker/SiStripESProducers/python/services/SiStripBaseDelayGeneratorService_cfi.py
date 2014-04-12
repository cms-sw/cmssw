# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms

SiStripBaseDelayGenerator = cms.Service("SiStripBaseDelayGenerator",
                                     file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
                                     CoarseDelay = cms.uint32(0),
                                     FineDelay = cms.uint32(0)
                                     )
