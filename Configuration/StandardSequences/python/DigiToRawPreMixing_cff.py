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

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith(DigiToRaw, DigiToRaw.copyAndExclude([castorRawData]))

#until we have hcal raw data for phase 2....
from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toReplaceWith(DigiToRaw, DigiToRaw.copyAndExclude([hcalRawData]))

# Remove siPixelRawData until we have phase1 pixel digis
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toReplaceWith(DigiToRaw, DigiToRaw.copyAndExclude([siPixelRawData])) # FIXME

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toReplaceWith(DigiToRaw, DigiToRaw.copyAndExclude([rpcpacker]))

from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen() :
    for _entry in [siPixelRawData,SiStripDigiToRaw,castorRawData]:
        DigiToRaw.remove(_entry)
