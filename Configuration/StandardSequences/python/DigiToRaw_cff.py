import FWCore.ParameterSet.Config as cms

# David Lange, Bryan Dahmes LLNL 
# February 26, 2007
#
# Definition of DigiToRaw sequence
#
# The DigiToRaw outputs from each subsystem are assembled into one collection
# (as is the case for data) to be read later by each subsystem's RawToDigi module.  
#-------------------------#
#--- DigiToRaw modules ---#
#-------------------------#
#--- CSCTF ---#
from EventFilter.CSCTFRawToDigi.csctfpacker_cfi import *
#--- DTTF ---#
from EventFilter.DTTFRawToDigi.dttfpacker_cfi import *
#--- GCT ---#
from EventFilter.GctRawToDigi.gctDigiToRaw_cfi import *
#--- GT ---#
from EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi import *
#--- SiPixel ---#
from EventFilter.SiPixelRawToDigi.SiPixelDigiToRaw_cfi import *
#--- SiStrip ---#
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
#--- Ecal ---#
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
import copy
from EventFilter.EcalDigiToRaw.ecalDigiToRaw_cfi import *
ecalPacker = copy.deepcopy(ecaldigitorawzerosup)
#--- Ecal Preshower ---#
from EventFilter.ESDigiToRaw.esDigiToRaw_cfi import *
#--- Hcal ---#
from EventFilter.HcalRawToDigi.HcalDigiToRaw_cfi import *
#--- CSC ---#
from EventFilter.CSCRawToDigi.cscPacker_cfi import *
#--- DT ---#
from EventFilter.DTRawToDigi.dtPacker_cfi import *
#--- RPC ---#
from EventFilter.RPCRawToDigi.rpcPacker_cfi import *
#--- Collect everything together ---#
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
DigiToRaw = cms.Sequence(csctfpacker*dttfpacker*gctDigiToRaw*l1GtPack*siPixelRawData*SiStripDigiToRaw*ecalPacker*esDigiToRaw*hcalRawData*cscpacker*dtpacker*rpcpacker*rawDataCollector)
ecalPacker.Label = 'ecalDigis'
ecalPacker.InstanceEB = 'ebDigis'
ecalPacker.InstanceEE = 'eeDigis'
ecalPacker.labelEBSRFlags = cms.InputTag("ecalDigis","ebSrFlags")
ecalPacker.labelEESRFlags = cms.InputTag("ecalDigis","eeSrFlags")

