import FWCore.ParameterSet.Config as cms

from DQMOffline.Lumi.ZCounting_cfi import *
zcounting = cms.Sequence(ZCounting)
