import FWCore.ParameterSet.Config as cms

# This object is used to selectively make changes for different running
# scenarios. In this case it makes changes for Run 2.

from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *

from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *

from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *

import EventFilter.EcalRawToDigi.EcalUnpackerData_cfi
ecalDigis = EventFilter.EcalRawToDigi.EcalUnpackerData_cfi.ecalEBunpacker.clone()

import EventFilter.ESRawToDigi.esRawToDigi_cfi
ecalPreshowerDigis = EventFilter.ESRawToDigi.esRawToDigi_cfi.esRawToDigi.clone()

import EventFilter.HcalRawToDigi.HcalRawToDigi_cfi
hcalDigis = EventFilter.HcalRawToDigi.HcalRawToDigi_cfi.hcalDigis.clone()

import EventFilter.CSCRawToDigi.cscUnpacker_cfi
muonCSCDigis = EventFilter.CSCRawToDigi.cscUnpacker_cfi.muonCSCDigis.clone()

import EventFilter.DTRawToDigi.dtunpacker_cfi
muonDTDigis = EventFilter.DTRawToDigi.dtunpacker_cfi.muonDTDigis.clone()

import EventFilter.RPCRawToDigi.rpcUnpacker_cfi
muonRPCDigis = EventFilter.RPCRawToDigi.rpcUnpacker_cfi.rpcunpacker.clone()

import EventFilter.RPCRawToDigi.rpcDigiMerger_cfi
muonRPCNewDigis = EventFilter.RPCRawToDigi.rpcDigiMerger_cfi.rpcDigiMerger.clone()

import EventFilter.GEMRawToDigi.muonGEMDigis_cfi
muonGEMDigis = EventFilter.GEMRawToDigi.muonGEMDigis_cfi.muonGEMDigis.clone()

from EventFilter.CastorRawToDigi.CastorRawToDigi_cff import *
castorDigis = EventFilter.CastorRawToDigi.CastorRawToDigi_cfi.castorDigis.clone( FEDs = cms.untracked.vint32(690,691,692, 693,722) )

from EventFilter.ScalersRawToDigi.ScalersRawToDigi_cfi import *

from EventFilter.Utilities.tcdsRawToDigi_cfi import *
tcdsDigis = EventFilter.Utilities.tcdsRawToDigi_cfi.tcdsRawToDigi.clone()

from EventFilter.OnlineMetaDataRawToDigi.onlineMetaDataRawToDigi_cfi import *
onlineMetaDataDigis = EventFilter.OnlineMetaDataRawToDigi.onlineMetaDataRawToDigi_cfi.onlineMetaDataRawToDigi.clone()

from L1Trigger.Configuration.L1TRawToDigi_cff import *

from EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff import *

RawToDigi = cms.Sequence(L1TRawToDigi
                         +siPixelDigis
                         +siStripDigis
                         +ecalDigis
                         +ecalPreshowerDigis
                         +hcalDigis
                         +muonCSCDigis
                         +muonDTDigis
                         +muonRPCDigis
                         +castorDigis
                         +scalersRawToDigi
                         +tcdsDigis
                         +onlineMetaDataDigis
                         )

RawToDigi_noTk = cms.Sequence(L1TRawToDigi
                              +ecalDigis
                              +ecalPreshowerDigis
                              +hcalDigis
                              +muonCSCDigis
                              +muonDTDigis
                              +muonRPCDigis
                              +castorDigis
                              +scalersRawToDigi
                              +tcdsDigis
                              +onlineMetaDataDigis
                              )

RawToDigi_pixelOnly = cms.Sequence(siPixelDigis)

scalersRawToDigi.scalersInputTag = 'rawDataCollector'
siPixelDigis.InputLabel = 'rawDataCollector'
#false by default anyways ecalDigis.DoRegional = False
ecalDigis.InputLabel = 'rawDataCollector'
ecalPreshowerDigis.sourceTag = 'rawDataCollector'
hcalDigis.InputLabel = 'rawDataCollector'
muonCSCDigis.InputObjects = 'rawDataCollector'
muonDTDigis.inputLabel = 'rawDataCollector'
muonRPCDigis.InputLabel = 'rawDataCollector'
castorDigis.InputLabel = 'rawDataCollector'

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toReplaceWith(RawToDigi, RawToDigi.copyAndExclude([castorDigis]))

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
# Remove siPixelDigis until we have phase1 pixel digis
phase2_tracker.toReplaceWith(RawToDigi, RawToDigi.copyAndExclude([siPixelDigis])) # FIXME


# add CTPPS 2016 raw-to-digi modules
from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016

_ctpps_2016_RawToDigi = RawToDigi.copy()
_ctpps_2016_RawToDigi += ctppsRawToDigi
ctpps_2016.toReplaceWith(RawToDigi, _ctpps_2016_RawToDigi)

_ctpps_2016_RawToDigi_noTk = RawToDigi_noTk.copy()
_ctpps_2016_RawToDigi_noTk += ctppsRawToDigi
ctpps_2016.toReplaceWith(RawToDigi_noTk, _ctpps_2016_RawToDigi_noTk)

# GEM settings
_gem_RawToDigi = RawToDigi.copy()
_gem_RawToDigi.insert(-1,muonGEMDigis)

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
run2_GEM_2017.toReplaceWith(RawToDigi, _gem_RawToDigi)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith(RawToDigi, _gem_RawToDigi)

from EventFilter.HGCalRawToDigi.HGCalRawToDigi_cfi import *
_hgcal_RawToDigi = RawToDigi.copy()
_hgcal_RawToDigi += hgcalDigis
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith(RawToDigi,_hgcal_RawToDigi)

# RPC New Readout Validation
from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017
_rpc_NewReadoutVal_RawToDigi = RawToDigi.copy()
_rpc_NewReadoutVal_RawToDigi_noTk = RawToDigi_noTk.copy()
_rpc_NewReadoutVal_RawToDigi += muonRPCNewDigis
_rpc_NewReadoutVal_RawToDigi_noTk += muonRPCNewDigis
stage2L1Trigger_2017.toReplaceWith(RawToDigi, _rpc_NewReadoutVal_RawToDigi)
stage2L1Trigger_2017.toReplaceWith(RawToDigi_noTk, _rpc_NewReadoutVal_RawToDigi)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(RawToDigi, RawToDigi.copyAndExclude([muonRPCNewDigis]))
fastSim.toReplaceWith(RawToDigi_noTk, RawToDigi_noTk.copyAndExclude([muonRPCNewDigis]))

_hfnose_RawToDigi = RawToDigi.copy()
_hfnose_RawToDigi += hfnoseDigis

from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
phase2_hfnose.toReplaceWith(RawToDigi,_hfnose_RawToDigi)

