import FWCore.ParameterSet.Config as cms

from DQM.SiStripCommon.TkHistoMap_cff import *
from CalibTracker.SiStripHitEfficiency.siStripHitEfficiencyHarvester_cfi import siStripHitEfficiencyHarvester
alcasiStripHitEfficiencyHarvester =  siStripHitEfficiencyHarvester.clone(
    isAtPCL             = True,
    Threshold           = 0.1,
    nModsMin            = 5,
    AutoIneffModTagging = True,  # default true, automatic limit for each layer to identify inefficient modules
    Record              = 'SiStripBadStripFromHitEffRcd',
    doStoreOnDB         = True,
    ShowRings           = False,  # default False
    TkMapMin            = 0.90, # default 0.90
    Title               = 'SiStrip Hit Efficiency Map')
