import FWCore.ParameterSet.Config as cms

#  description:
# workflow for L1 Trigger Emulator DQM
# used by DQM GUI: DQM/Integration/l1temulator*.cfg
# nuno.leonardo@cern.ch 08.02
# 
#systems configuration
from L1Trigger.HardwareValidation.hwtest.globrun.deGmt_cff import *
#include "L1Trigger/HardwareValidation/hwtest/globrun/deRpc.cff"
#include "L1Trigger/HardwareValidation/hwtest/globrun/deDt.cff"
#include "L1Trigger/HardwareValidation/hwtest/globrun/deCsc.cff"
#include "L1Trigger/HardwareValidation/hwtest/globrun/deGct.cff"
#include "L1Trigger/HardwareValidation/hwtest/globrun/deRct.cff"
#include "L1Trigger/HardwareValidation/hwtest/globrun/deEcal.cff"
#include "L1Trigger/HardwareValidation/hwtest/globrun/deHcal.cff"
#data|emulator comparator
from L1Trigger.HardwareValidation.L1Comparator_cfi import *
#sequence {ETP,HTP,RCT,GCT, DTP,DTF,CTP,CTF,RPC, LTC,GMT,GT}
#dqm sources    
from DQM.L1TMonitor.L1TDEMON_cfi import *
#calo unpack
#path unpketp = {ecalEBunpacker}
#path unpkhtp = {hcalDigis}
#path unpkgct = {gctUnpacker}
#calo emulator
#path emuletp = {ecalTriggerPrimitiveDigis}
#path emulhtp = {hcalTriggerPrimitiveDigis}
#path emulrct = {maskedRctInputDigis, l1RctEmulDigis }
#path emulgct = {l1GctEmulDigis}
#muon unpack
#path unpkctp = {muonCSCDigis}
#path unpkctf = {csctfunpacker}
#path unpkdtp = {muonDTDigis}
#path unpkdtf = {muonDTTFDigis}
#path unpkrpc = {muonRPCDigis}
unpkglt = cms.Path(gtUnpack)
#muon emulator
#path emulctp = {l1CscTpgEmulDigis}
#path emulctf = {l1CscTfTrackEmulDigis,l1CscTfEmulDigis}
#path emuldtp = {l1DTTPGEmulDigis}
#path emuldtf = {l1DttfEmulDigis}
#path emulrpc = {l1RpcEmulDigis}
emulgmt = cms.Path(cms.SequencePlaceholder("l1GmtEmulDigis"))
#comparator, dqm
anacomp = cms.Path(l1compare*l1demon)
#ecal
l1compare.ETPsourceData = cms.InputTag("ecalEBunpacker","EBTT")
l1compare.ETPsourceEmul = 'ecalTriggerPrimitiveDigis'
#hcal
l1compare.HTPsourceData = 'hcalDigis'
l1compare.HTPsourceEmul = 'hcalTriggerPrimitiveDigis'
#rct
l1compare.RCTsourceData = 'gctUnpacker'
l1compare.RCTsourceEmul = 'l1RctEmulDigis'
#gct
l1compare.GCTsourceData = 'gctUnpacker'
l1compare.GCTsourceEmul = 'l1GctEmulDigis'
#dt
l1compare.DTPsourceData = 'muonDTTFDigis'
l1compare.DTPsourceEmul = '1DTTPGEmulDigis'
l1compare.DTFsourceData = 'muonDTTFDigis'
l1compare.DTFsourceEmul = 'l1DttfEmulDigis'
#csc
l1compare.CTPsourceData = 'csctfunpacker'
l1compare.CTPsourceEmul = 'l1CscTpgEmulDigis'
l1compare.CTFsourceData = 'csctfunpacker'
l1compare.CTFsourceEmul = 'l1CscTfEmulDigis'
l1compare.CTTsourceData = 'csctfunpacker'
l1compare.CTTsourceEmul = 'l1CscTfTrackEmulDigis'
#rpc
l1compare.RPCsourceData = 'muonRPCDigis'
l1compare.RPCsourceEmul = 'l1RpcEmulDigis'
#gmt
l1compare.GMTsourceData = 'gtUnpack'
l1compare.GMTsourceEmul = 'l1GmtEmulDigis'
l1compare.COMPARE_COLLS = [0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 
    1, 0]

