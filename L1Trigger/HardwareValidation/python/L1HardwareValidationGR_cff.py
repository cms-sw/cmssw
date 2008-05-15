import FWCore.ParameterSet.Config as cms

import SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi
# L1 Emulator-Hardware comparison sequences -- Global Run
#
# J. Brooke, N. Leonardo
#
# These sequences assume RawToDigi has run
# Note that the emulator configuration also needs to be supplied
# Either from dummy ES producers, or DB
# ECAL sequence
# requires ecalDigis only
valEcalTriggerPrimitiveDigis = SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi.simEcalTriggerPrimitiveDigis.clone()
import SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cfi
# HCAL sequence
# requires hcalDigis only
valHcalTriggerPrimitiveDigis = SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cfi.simHcalTriggerPrimitiveDigis.clone()
# RCT sequence
# requires ecalDigis and hcalDigis
from L1Trigger.RegionalCaloTrigger.maskedRctInputDigis_cfi import *
import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
valRctDigis = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone()
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
# GCT sequence
# requires gctDigis
valGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
import L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi
# DT TPG sequence
# requires muonDTDigis only
valDtTriggerPrimitiveDigis = L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi.dtTriggerPrimitiveDigis.clone()
import L1Trigger.DTTrackFinder.dttfDigis_cfi
# DT TF sequence
# requires dttfDigis and csctfDigis
# currently generates CSCTF stubs by running CSCTF emulator
valDttfDigis = L1Trigger.DTTrackFinder.dttfDigis_cfi.dttfDigis.clone()
import L1Trigger.HardwareValidation.MuonCandProducerMon_cfi
#replace valDttfDigis.CSCStub_Source = csctfDigis
muonDtMon = L1Trigger.HardwareValidation.MuonCandProducerMon_cfi.muonCandMon.clone()
import L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi
# CSC TPG sequence
# requires muonCSCDigis only
valCscTriggerPrimitiveDigis = L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi.cscTriggerPrimitiveDigis.clone()
import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi
# CSC TF sequence
# requires csctfDigis and dttfDigis
valCsctfTrackDigis = L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi.csctfTrackDigis.clone()
import L1Trigger.CSCTrackFinder.csctfDigis_cfi
valCsctfDigis = L1Trigger.CSCTrackFinder.csctfDigis_cfi.csctfDigis.clone()
import L1Trigger.HardwareValidation.MuonCandProducerMon_cfi
muonCscMon = L1Trigger.HardwareValidation.MuonCandProducerMon_cfi.muonCandMon.clone()
import L1Trigger.RPCTrigger.rpcTriggerDigis_cfi
# RPC sequence
# requires muonRPCDigis only
valRpcTriggerDigis = L1Trigger.RPCTrigger.rpcTriggerDigis_cfi.rpcTriggerDigis.clone()
import L1Trigger.GlobalMuonTrigger.gmtDigis_cfi
# GMT sequence
# requires gtDigis only
valGmtDigis = L1Trigger.GlobalMuonTrigger.gmtDigis_cfi.gmtDigis.clone()
import L1Trigger.GlobalTrigger.gtDigis_cfi
# GT sequence
# requires gtDigis and gctDigis
valGtDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()
# the comparator module
from L1Trigger.HardwareValidation.L1Comparator_cfi import *
deEcal = cms.Sequence(valEcalTriggerPrimitiveDigis)
deHcal = cms.Sequence(valHcalTriggerPrimitiveDigis)
deRct = cms.Sequence(maskedRctInputDigis*valRctDigis)
deGct = cms.Sequence(valGctDigis)
#replace valDtTriggerPrimitiveDigis.inputLabel = muonDTDigis
deDt = cms.Sequence(valDtTriggerPrimitiveDigis)
deDttf = cms.Sequence(valCsctfTrackDigis*valDttfDigis*muonDtMon)
deCsc = cms.Sequence(valCscTriggerPrimitiveDigis)
deCsctf = cms.Sequence(valCsctfTrackDigis*valCsctfDigis*muonCscMon)
deRpc = cms.Sequence(valRpcTriggerDigis)
#replace valGmtDigis.MipIsoData =
deGmt = cms.Sequence(valGmtDigis)
#replace valGtDigis.TechnicalTriggerInputTag = ???
deGt = cms.Sequence(valGtDigis)
#replace l1compare.COMPARE_COLLS= { 0,0,0,0,0,0,0,0,0,0,0,0 }
#compareMode  {ETP,HTP,RCT,GCT, DTP,DTF,CTP,CTF,RPC, LTC,GMT,GT};
L1HardwareValidation = cms.Sequence(deEcal+deHcal+deRct+deGct+deDt+deDttf+deCsc+deCsctf+deRpc+deGmt+deGt*l1compare)
valEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
valEcalTriggerPrimitiveDigis.InstanceEB = 'ebDigis'
valEcalTriggerPrimitiveDigis.InstanceEE = 'eeDigis'
valHcalTriggerPrimitiveDigis.inputLabel = 'hcalDigis'
#cruzet-i: no ecal tp's, need emulation instead:
maskedRctInputDigis.ecalDigisLabel = 'valEcalTriggerPrimitiveDigis'
maskedRctInputDigis.hcalDigisLabel = 'hcalDigis'
valRctDigis.ecalDigisLabel = 'maskedRctInputDigis'
valRctDigis.hcalDigisLabel = 'maskedRctInputDigis'
valGctDigis.inputLabel = 'gctDigis'
valDttfDigis.DTDigi_Source = 'dttfDigis'
valDttfDigis.CSCStub_Source = 'valCsctfTrackDigis'
muonDtMon.CSCinput = 'dttfDigis'
valCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi")
valCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag("muonCSCDigis","MuonCSCWireDigi")
valCsctfTrackDigis.SectorReceiverInput = 'csctfDigis'
#replace valCsctfTrackDigis.useDT               = false
valCsctfTrackDigis.DTproducer = 'dttfDigis'
valCsctfDigis.CSCTrackProducer = 'valCsctfTrackDigis'
muonCscMon.CSCinput = 'csctfDigis'
valRpcTriggerDigis.label = 'muonRPCDigis'
valGmtDigis.DTCandidates = cms.InputTag("gtDigis","DT")
valGmtDigis.CSCCandidates = cms.InputTag("gtDigis","CSC")
valGmtDigis.RPCbCandidates = cms.InputTag("gtDigis","RPCb")
valGmtDigis.RPCfCandidates = cms.InputTag("gtDigis","RPCf")
valGtDigis.GmtInputTag = 'gtDigis'
valGtDigis.GctInputTag = 'gctDigis'
#note: cruzet-i, no tp's readout
l1compare.ETPsourceData = 'valEcalTriggerPrimitiveDigis'
l1compare.ETPsourceEmul = 'valEcalTriggerPrimitiveDigis'
l1compare.HTPsourceData = 'hcalDigis'
l1compare.HTPsourceEmul = 'valHcalTriggerPrimitiveDigis'
l1compare.RCTsourceData = 'gctDigis'
l1compare.RCTsourceEmul = 'valRctDigis'
l1compare.GCTsourceData = 'gctDigis'
l1compare.GCTsourceEmul = 'valGctDigis'
l1compare.DTPsourceData = 'dttfDigis' ##muonDTDigis

l1compare.DTPsourceEmul = 'valDtTriggerPrimitiveDigis'
l1compare.DTFsourceData = cms.InputTag("muonDtMon","DT") ##dttfDigis

l1compare.DTFsourceEmul = 'valDttfDigis'
#csc tp comparison 
# i: csctf-readout vs csc-emulator
l1compare.CTPsourceData = 'csctfDigis'
l1compare.CTPsourceEmul = cms.InputTag("valCscTriggerPrimitiveDigis","MPCSORTED")
# ii:csc-readout vs csc-emulator
#replace l1compare.CTPsourceData = muonCSCDigis:MuonCSCCorrelatedLCTDigi
#replace l1compare.CTPsourceEmul = valCscTriggerPrimitiveDigis
l1compare.CTFsourceData = cms.InputTag("muonCscMon","CSC")
l1compare.CTFsourceEmul = cms.InputTag("valCsctfDigis","CSC")
l1compare.CTTsourceData = 'csctfDigis'
l1compare.CTTsourceEmul = 'valCsctfTrackDigis'
l1compare.RPCsourceData = 'gtDigis'
l1compare.RPCsourceEmul = 'valRpcTriggerDigis'
l1compare.GMTsourceData = 'gtDigis'
l1compare.GMTsourceEmul = 'valGmtDigis'
l1compare.GLTsourceData = 'gtDigis'
l1compare.GLTsourceEmul = 'valGtDigis'


