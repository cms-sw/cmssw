import FWCore.ParameterSet.Config as cms

# L1 Emulator-Hardware comparison sequences -- Global Run
#
# J. Brooke, N. Leonardo
#
# included here:
# - emulator sequences
# - specification of emulator inputs
# - definition of validation sequences
#
# these sequences assume RawToDigi has run
#
# note that the emulator configuration also needs to be supplied
#  either from dummy ES producers, or DB
# note fake conditions for ECAL/HCAL - should be moved out of here
#  and into Fake/FrontierConditions

from SimCalorimetry.EcalTrigPrimProducers.ecalTrigPrimESProducer_cff import *
import SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi
valEcalTriggerPrimitiveDigis = SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cfi.simEcalTriggerPrimitiveDigis.clone()

# HCAL sequence
# requires hcalDigis only
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
valHcalTriggerPrimitiveDigis = SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cfi.simHcalTriggerPrimitiveDigis.clone()

# RCT sequence
# requires ecalDigis and hcalDigis
import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
valRctDigis = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone()

# GCT sequence
# requires gctDigis
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
valGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()

# DT TPG sequence
# requires muonDTDigis only
#import L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi
from L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi import *
valDtTriggerPrimitiveDigis = L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi.dtTriggerPrimitiveDigis.clone()

# DT TF sequence
# requires dttfDigis and csctfDigis
# currently generates CSCTF stubs by running CSCTF emulator
import L1Trigger.DTTrackFinder.dttfDigis_cfi
valDttfDigis = L1Trigger.DTTrackFinder.dttfDigis_cfi.dttfDigis.clone()
#replace valDttfDigis.CSCStub_Source = csctfDigis
import L1Trigger.HardwareValidation.MuonCandProducerMon_cfi
muonDtMon = L1Trigger.HardwareValidation.MuonCandProducerMon_cfi.muonCandMon.clone()

# CSC TPG sequence
# requires muonCSCDigis only
import L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi
valCscTriggerPrimitiveDigis = L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi.cscTriggerPrimitiveDigis.clone()

# CSC TF sequence
# requires csctfDigis and dttfDigis
import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi
valCsctfTrackDigis = L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi.csctfTrackDigis.clone()
import L1Trigger.CSCTrackFinder.csctfDigis_cfi
valCsctfDigis = L1Trigger.CSCTrackFinder.csctfDigis_cfi.csctfDigis.clone()
import L1Trigger.HardwareValidation.MuonCandProducerMon_cfi
muonCscMon = L1Trigger.HardwareValidation.MuonCandProducerMon_cfi.muonCandMon.clone()

# RPC sequence
# requires muonRPCDigis only
#import L1Trigger.RPCTrigger.rpcTriggerDigis_cfi
from L1Trigger.RPCTrigger.rpcTriggerDigis_cff import *
valRpcTriggerDigis = L1Trigger.RPCTrigger.rpcTriggerDigis_cfi.rpcTriggerDigis.clone()

# GMT sequence
# requires gtDigis only
import L1Trigger.GlobalMuonTrigger.gmtDigis_cfi
valGmtDigis = L1Trigger.GlobalMuonTrigger.gmtDigis_cfi.gmtDigis.clone()

# GT sequence
# requires gtDigis and gctDigis
import L1Trigger.GlobalTrigger.gtDigis_cfi
valGtDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()

# Emulator input
valEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
valEcalTriggerPrimitiveDigis.InstanceEB = 'ebDigis'
valEcalTriggerPrimitiveDigis.InstanceEE = 'eeDigis'
valHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(cms.InputTag('hcalDigis'),cms.InputTag('hcalDigis'))
valRctDigis.ecalDigis = cms.VInputTag(cms.InputTag('ecalDigis:EcalTriggerPrimitives'))
valRctDigis.hcalDigis = cms.VInputTag(cms.InputTag('hcalDigis'))
valGctDigis.inputLabel = 'gctDigis'
#valDtTriggerPrimitiveDigis.inputLabel = 'muonDTDigis'
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
valGmtDigis.MipIsoData = 'gctDigis'
#replace valGmtDigis.MipIsoData =
#replace valGtDigis.TechnicalTriggerInputTag = ???

#Emulator settings
valHcalTriggerPrimitiveDigis.FG_threshold = cms.uint32(12)
EcalTrigPrimESProducer.DatabaseFile = 'TPG_startup.txt.gz'
HcalTPGCoderULUT.read_Ascii_LUTs = True
HcalTPGCoderULUT.inputLUTs = 'L1Trigger/HardwareValidation/hwtest/globrun/HcalCRAFTPhysicsV2.dat'

# the comparator module
# parameters are specified in cfi
from L1Trigger.HardwareValidation.L1Comparator_cfi import *
#l1compare.COMPARE_COLLS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
# ETP,HTP,RCT,GCT, DTP,DTF,CTP,CTF,RPC, LTC,GMT,GT

# subsystem sequences
deEcal = cms.Sequence(valEcalTriggerPrimitiveDigis)
deHcal = cms.Sequence(valHcalTriggerPrimitiveDigis)
deRct = cms.Sequence(valRctDigis)
deGct = cms.Sequence(valGctDigis)
deDt = cms.Sequence(valDtTriggerPrimitiveDigis)
deDttf = cms.Sequence(valCsctfTrackDigis*valDttfDigis*muonDtMon)
deCsc = cms.Sequence(valCscTriggerPrimitiveDigis)
deCsctf = cms.Sequence(valCsctfTrackDigis*valCsctfDigis*muonCscMon)
deRpc = cms.Sequence(valRpcTriggerDigis)
deGmt = cms.Sequence(valGmtDigis)
deGt = cms.Sequence(valGtDigis)

# the sequences
L1HardwareValidation = cms.Sequence(deEcal+deHcal+deRct+deGct+deDt+deDttf+deCsc+deCsctf+deRpc+deGmt+deGt*l1compare)
