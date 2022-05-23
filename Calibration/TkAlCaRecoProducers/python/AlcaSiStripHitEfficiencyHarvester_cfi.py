import copy
import FWCore.ParameterSet.Config as cms

from DQM.SiStripCommon.TkHistoMap_cff import *
from CalibTracker.SiStripHitEfficiency.siStripHitEfficiencyHarvester_cfi import siStripHitEfficiencyHarvester
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdSiStripHitEfficiency_cff import ALCARECOSiStripHitEff

alcasiStripHitEfficiencyHarvester =  siStripHitEfficiencyHarvester.clone(
    inputFolder         = copy.copy(ALCARECOSiStripHitEff.dqmDir), # make sure the harvester is always in synch with worker
    isAtPCL             = True,
    Threshold           = 0.1,
    nModsMin            = 5,
    AutoIneffModTagging = True,  # default true, automatic limit for each layer to identify inefficient modules
    Record              = 'SiStripBadStripFromHitEffRcd',
    doStoreOnDB         = True,
    ShowRings           = False,  # default False
    TkMapMin            = 0.90, # default 0.90
    Title               = 'SiStrip Hit Efficiency Map')
