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

#DigiToRaw = cms.Sequence(L1TDigiToRaw*siPixelRawData*SiStripDigiToRaw*ecalPacker*esDigiToRaw*hcalRawData*cscpacker*dtpacker*rpcpacker*rawDataCollector)
DigiToRaw = cms.Sequence(L1TDigiToRaw*siPixelRawData*SiStripDigiToRaw*ecalPacker*esDigiToRaw*hcalRawData*cscpacker*dtpacker*rpcpacker*castorRawData*rawDataCollector)

ecalPacker.Label = 'simEcalDigis'
ecalPacker.InstanceEB = 'ebDigis'
ecalPacker.InstanceEE = 'eeDigis'
ecalPacker.labelEBSRFlags = "simEcalDigis:ebSrFlags"
ecalPacker.labelEESRFlags = "simEcalDigis:eeSrFlags"

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toReplaceWith(DigiToRaw, DigiToRaw.copyAndExclude([castorRawData]))

#if we don't have hcal raw data
from Configuration.Eras.Modifier_hcalSkipPacker_cff import hcalSkipPacker
hcalSkipPacker.toReplaceWith(DigiToRaw, DigiToRaw.copyAndExclude([hcalRawData]))

# Remove siPixelRawData until we have phase1 pixel digis
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(DigiToRaw, DigiToRaw.copyAndExclude([siPixelRawData])) # FIXME

# GEM settings
_gem_DigiToRaw = DigiToRaw.copy()
_gem_DigiToRaw.insert(-2,gemPacker)

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toReplaceWith(DigiToRaw, _gem_DigiToRaw)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith(DigiToRaw, _gem_DigiToRaw)

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith(DigiToRaw, DigiToRaw.copyAndExclude([rpcpacker]))

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(DigiToRaw, DigiToRaw.copyAndExclude([siPixelRawData,SiStripDigiToRaw,castorRawData]))
