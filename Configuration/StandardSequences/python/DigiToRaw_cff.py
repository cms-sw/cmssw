import FWCore.ParameterSet.Config as cms

# This object is used to make changes for different running scenarios. In
# this case for Run 2

from EventFilter.SiPixelRawToDigi.SiPixelDigiToRaw_cfi import *
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
import EventFilter.EcalDigiToRaw.ecalDigiToRaw_cfi
ecalPacker = EventFilter.EcalDigiToRaw.ecalDigiToRaw_cfi.ecaldigitorawzerosup.clone()
from EventFilter.ESDigiToRaw.esDigiToRaw_cfi import *
from EventFilter.HcalRawToDigi.HcalDigiToRaw_cfi import *
from EventFilter.CSCRawToDigi.cscPacker_cfi import *
from EventFilter.DTRawToDigi.dtPacker_cfi import *
from EventFilter.RPCRawToDigi.rpcPacker_cfi import *
from EventFilter.GEMRawToDigi.gemPacker_cfi import *
from EventFilter.CastorRawToDigi.CastorDigiToRaw_cfi import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
from L1Trigger.Configuration.L1TDigiToRaw_cff import *
from EventFilter.CTPPSRawToDigi.ctppsDigiToRaw_cff import *

DigiToRawTask = cms.Task(L1TDigiToRawTask, siPixelRawData, SiStripDigiToRaw, ecalPacker, esDigiToRaw, hcalRawDataTask, cscpacker, dtpacker, rpcpacker, ctppsRawData, castorRawData, rawDataCollector)
DigiToRaw = cms.Sequence(DigiToRawTask)

ecalPacker.Label = 'simEcalDigis'
ecalPacker.InstanceEB = 'ebDigis'
ecalPacker.InstanceEE = 'eeDigis'
ecalPacker.labelEBSRFlags = "simEcalDigis:ebSrFlags"
ecalPacker.labelEESRFlags = "simEcalDigis:eeSrFlags"

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toReplaceWith(DigiToRawTask, DigiToRawTask.copyAndExclude([castorRawData]))

#if we don't have hcal raw data
from Configuration.Eras.Modifier_hcalSkipPacker_cff import hcalSkipPacker
hcalSkipPacker.toReplaceWith(DigiToRawTask, DigiToRawTask.copyAndExclude([hcalRawDataTask]))

# Remove siPixelRawData until we have phase1 pixel digis
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(DigiToRawTask, DigiToRawTask.copyAndExclude([siPixelRawData])) # FIXME

# GEM settings
_gem_DigiToRawTask = DigiToRawTask.copy()
_gem_DigiToRawTask.add(gemPacker)

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toReplaceWith(DigiToRawTask, _gem_DigiToRawTask)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith(DigiToRawTask, _gem_DigiToRawTask)

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith(DigiToRawTask, DigiToRawTask.copyAndExclude([rpcpacker]))

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(DigiToRawTask, DigiToRawTask.copyAndExclude([siPixelRawData,SiStripDigiToRaw,castorRawData,ctppsRawData]))

from Configuration.Eras.Modifier_phase2_ecal_devel_cff import phase2_ecal_devel
phase2_ecal_devel.toReplaceWith(DigiToRawTask, DigiToRawTask.copyAndExclude([esDigiToRaw]))
