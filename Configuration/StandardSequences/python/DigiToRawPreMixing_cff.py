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
from EventFilter.GEMRawToDigi.me0Packer_cfi import *
from EventFilter.CastorRawToDigi.CastorDigiToRaw_cfi import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
#from L1Trigger.Configuration.L1TDigiToRaw_cff import *  # no L1 DigiToRaw in first PreMixing step
#DigiToRaw = cms.Sequence(L1TDigiToRaw*siPixelRawData*SiStripDigiToRaw*ecalPacker*esDigiToRaw*hcalRawData*cscpacker*dtpacker*rpcpacker*rawDataCollector)
DigiToRaw = cms.Sequence(siPixelRawData*SiStripDigiToRaw*ecalPacker*esDigiToRaw*hcalRawData*cscpacker*dtpacker*rpcpacker*castorRawData*rawDataCollector)
ecalPacker.Label = 'simEcalDigis'
ecalPacker.InstanceEB = 'ebDigis'
ecalPacker.InstanceEE = 'eeDigis'
ecalPacker.labelEBSRFlags = "simEcalDigis:ebSrFlags"
ecalPacker.labelEESRFlags = "simEcalDigis:eeSrFlags"
hcalRawDatauHTR.premix = cms.bool(True)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toReplaceWith(DigiToRaw, DigiToRaw.copyAndExclude([castorRawData]))

#until we have hcal raw data for phase 2....
from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toReplaceWith(DigiToRaw, DigiToRaw.copyAndExclude([hcalRawData]))

# Remove siPixelRawData until we have phase1 pixel digis
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(DigiToRaw, DigiToRaw.copyAndExclude([siPixelRawData])) # FIXME

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith(DigiToRaw, DigiToRaw.copyAndExclude([rpcpacker]))

# GEM settings
_run2_GEM_2017_DigiToRaw = DigiToRaw.copy()
_run2_GEM_2017_DigiToRaw.insert(0,gempacker)

_run3_DigiToRaw = _run2_GEM_2017_DigiToRaw.copy()
#_run3_DigiToRaw.insert(0,me0packer) # to be done once proper ME0 digitiser is done

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toReplaceWith(DigiToRaw, _run2_GEM_2017_DigiToRaw)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith(DigiToRaw, _run3_DigiToRaw)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen() :
    for _entry in [siPixelRawData,SiStripDigiToRaw,castorRawData]:
        DigiToRaw.remove(_entry)
