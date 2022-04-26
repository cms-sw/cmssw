import FWCore.ParameterSet.Config as cms

from DQM.SiStripCommon.TkHistoMap_cff import *
from CalibTracker.SiStripHitEfficiency.siStripHitEfficiencyHarvester_cfi import siStripHitEfficiencyHarvester
alcasiStripHitEfficiencyHarvester =  siStripHitEfficiencyHarvester.clone(
    isAtPCL             = cms.bool(True),
    Threshold           = cms.double(0.1),
    nModsMin            = cms.int32(5),
    AutoIneffModTagging = cms.untracked.bool(True),  # default true, automatic limit for each layer to identify inefficient modules
    Record              = cms.string('SiStripBadStripFromHitEffRcd'),
    doStoreOnDB         = cms.bool(True),
    ShowRings           = cms.untracked.bool(False),  # default False
    TkMapMin            = cms.untracked.double(0.90), # default 0.90
    #EffPlotMin          = cms.untracked.double(0.90), # default 0.90
    Title               = cms.string('SiStrip Hit Efficiency Map'))
