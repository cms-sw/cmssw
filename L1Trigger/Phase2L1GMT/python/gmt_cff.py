import FWCore.ParameterSet.Config as cms
from L1Trigger.Phase2L1GMT.gmt_cfi import *
phase2GMT = cms.Sequence(gmtStubs*gmtMuons)
