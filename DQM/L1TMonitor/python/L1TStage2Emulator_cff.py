import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# Stage2 Emulator Modules (TODO: Move to L1Trigger.HardwareValidation.L1Stage2HardwareValidation_cff)

# Calo configuration
from L1Trigger.L1TCalorimeter.simDigis_cff import *
# CaloLayer1
from L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Digis_cfi import simCaloStage2Layer1Digis
valCaloStage2Layer1Digis = simCaloStage2Layer1Digis.clone(
    ecalToken = "caloLayer1Digis",
    hcalToken = "caloLayer1Digis",
    unpackEcalMask = True,
    unpackHcalMask = True
)

# CaloLayer2
from L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi import simCaloStage2Digis
valCaloStage2Layer2Digis = simCaloStage2Digis.clone(towerToken = "caloStage2Digis:CaloTower")

# BMTF-Legacy
from L1Trigger.L1TMuonBarrel.simBmtfDigis_cfi import *
valBmtfDigis = simBmtfDigis.clone(
    DTDigi_Source = "bmtfDigis",
    DTDigi_Theta_Source = "bmtfDigis"
)

# BMTF-Kalman
from L1Trigger.L1TMuonBarrel.simKBmtfDigis_cfi import *
from L1Trigger.L1TMuonBarrel.simKBmtfStubs_cfi import *
valKBmtfStubs = simKBmtfStubs.clone(
    srcPhi = "bmtfDigis",
    srcTheta = "bmtfDigis"
)
valKBmtfDigis = simKBmtfDigis.clone(src = "valKBmtfStubs")

# BMTF-AlgoTriggerSelector
from DQM.L1TMonitor.L1TBMTFAlgoSelector_cfi import *
valBmtfAlgoSel = l1tBmtfAlgoSelector.clone(
    bmtfKalman = "valKBmtfDigis:BMTF",
    bmtfLegacy = "valBmtfDigis:BMTF"
)

# OMTF
from L1Trigger.L1TMuonOverlapPhase1.simOmtfDigis_cfi import *
valOmtfDigis = simOmtfDigis.clone(
    srcDTPh = "omtfStage2Digis",
    srcDTTh = "omtfStage2Digis",
    srcCSC = "omtfStage2Digis",
    srcRPC = "omtfStage2Digis",
	bxMin = -3,
    bxMax = 4
)

# GEM TPG
from L1Trigger.L1TGEM.simGEMDigis_cff import *
valMuonGEMPadDigis = simMuonGEMPadDigis.clone(InputCollection = "muonGEMDigis")
valMuonGEMPadDigiClusters = simMuonGEMPadDigiClusters.clone(InputCollection = "valMuonGEMPadDigis")

# Lookup tables for the CSC TP emulator
from CalibMuon.CSCCalibration.CSCL1TPLookupTableEP_cff import *
# CSC TPG
from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi import *
valCscStage2Digis = cscTriggerPrimitiveDigis.clone(
    CSCComparatorDigiProducer = "muonCSCDigis:MuonCSCComparatorDigi",
    CSCWireDigiProducer = "muonCSCDigis:MuonCSCWireDigi",
    GEMPadDigiClusterProducer = "",
    commonParam = dict(runME11ILT = False)
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify( valCscStage2Digis,
                      keepShowers = True,
                      commonParam = dict(run3 = True,
                                         runCCLUT_OTMB = True,
                                         runPhase2 = True,
                                         runME11Up = True,
                                         runME21Up = True,
                                         runME31Up = True,
                                         runME41Up = True)
)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toModify( valCscStage2Digis, GEMPadDigiClusterProducer = "valMuonGEMPadDigiClusters" , commonParam = dict(runME11ILT = True) )

# CPPF
from RecoLocalMuon.RPCRecHit.rpcRecHits_cfi import *
valRpcRecHits = rpcRecHits.clone(
	rpcDigiLabel = 'rpcunpacker'
)
from L1Trigger.L1TMuonCPPF.emulatorCppfDigis_cfi import *
valCppfStage2Digis = emulatorCppfDigis.clone(
	recHitLabel = 'valRpcRecHits'
)

# EMTF
from L1Trigger.L1TMuonEndCap.simEmtfDigis_cfi import *
valEmtfStage2Digis = simEmtfDigis.clone(
    CSCInput = "emtfStage2Digis",
    RPCInput = "muonRPCDigis",
    GEMInput = 'valMuonGEMPadDigiClusters'
)
run3_GEM.toModify( valEmtfStage2Digis, UseRun3CCLUT_OTMB = cms.bool(True), Era = cms.string('Run3_2021'))

# EMTF shower
from L1Trigger.L1TMuonEndCap.simEmtfShowers_cfi import *
valEmtfStage2Showers = simEmtfShowers.clone(
    CSCShowerInput = cms.InputTag('valCscStage2Digis')
)

# uGMT
from L1Trigger.L1TMuon.simGmtStage2Digis_cfi import *
valGmtCaloSumDigis = simGmtCaloSumDigis.clone(caloStage2Layer2Label = "valCaloStage2Layer1Digis")
valGmtStage2Digis = simGmtStage2Digis.clone(
    barrelTFInput = "gmtStage2Digis:BMTF",
    overlapTFInput = "gmtStage2Digis:OMTF",
    forwardTFInput = "gmtStage2Digis:EMTF",
    triggerTowerInput = "valGmtCaloSumDigis:TriggerTowerSums"
)

# uGMT shower
valGmtShowerDigis = simGmtShowerDigis.clone(
    showerInput = "gmtStage2Digis:EMTF"
)

# uGT
from L1Trigger.L1TGlobal.simGtStage2Digis_cfi import simGtStage2Digis
from L1Trigger.L1TGlobal.simGtExtFakeProd_cfi import simGtExtFakeProd

valGtStage2Digis = simGtStage2Digis.clone(
    ExtInputTag = "gtStage2Digis",
    MuonInputTag = "gtStage2Digis:Muon",
    MuonShowerInputTag = "gtStage2Digis:MuonShower",
    EGammaInputTag = "gtStage2Digis:EGamma",
    TauInputTag = "gtStage2Digis:Tau",
    JetInputTag = "gtStage2Digis:Jet",
    EtSumInputTag = "gtStage2Digis:EtSum",
    EtSumZdcInputTag = "gtStage2Digis:EtSumZDC",
    AlgorithmTriggersUnmasked = False,
    AlgorithmTriggersUnprescaled = False,
    EmulateBxInEvent = cms.int32(5),
    PrescaleSet = cms.uint32(7),
    GetPrescaleColumnFromData = True,
    AlgoBlkInputTag = "gtStage2Digis"
)
Stage2L1HardwareValidation = cms.Sequence(
    valCaloStage2Layer1Digis +
    valCscStage2Digis +
    valBmtfDigis +
    valKBmtfStubs +
    valKBmtfDigis +
    valBmtfAlgoSel +
    valOmtfDigis +
    valRpcRecHits +
    valCppfStage2Digis +
    valEmtfStage2Digis +
    valGmtCaloSumDigis +
    valGmtStage2Digis +
    valGtStage2Digis
)

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
_run3_Stage2L1HardwareValidation = Stage2L1HardwareValidation.copy()
run3_GEM.toReplaceWith( Stage2L1HardwareValidation, cms.Sequence( valMuonGEMPadDigis + valMuonGEMPadDigiClusters + _run3_Stage2L1HardwareValidation) )

_run3Shower_Stage2L1HardwareValidation = Stage2L1HardwareValidation.copy()
run3_GEM.toReplaceWith( Stage2L1HardwareValidation, cms.Sequence(_run3Shower_Stage2L1HardwareValidation + valEmtfStage2Showers + valGmtShowerDigis) )

Stage2L1HardwareValidationForValidationEvents = cms.Sequence(
    valCaloStage2Layer2Digis
)

#-------------------------------------------------
# Emulator DQM Modules

# CaloLayer1
from DQM.L1TMonitor.L1TdeStage2CaloLayer1_cfi import *

# CaloLayer2
from DQM.L1TMonitor.L1TdeStage2CaloLayer2_cfi import *
from DQM.L1TMonitor.L1TStage2CaloLayer2Emul_cfi import *

# GEM TPG
from DQM.L1TMonitor.L1TdeGEMTPG_cfi import *

# CSC TPG
from DQM.L1TMonitor.L1TdeCSCTPG_cfi import *
from DQM.L1TMonitor.L1TdeCSCTPGShower_cfi import *

# BMTF
from DQM.L1TMonitor.L1TdeStage2BMTF_cfi import *
from DQM.L1TMonitor.L1TdeStage2BMTFSecond_cff import *

# OMTF
from DQM.L1TMonitor.L1TdeStage2OMTF_cfi import *

# CPPF
from DQM.L1TMonitor.L1TdeStage2CPPF_cff import *

# EMTF
from DQM.L1TMonitor.L1TdeStage2EMTF_cff import *

# uGMT
from DQM.L1TMonitor.L1TdeStage2uGMT_cff import *

# uGT
from DQM.L1TMonitor.L1TStage2uGTEmul_cfi import *
from DQM.L1TMonitor.L1TdeStage2uGT_cfi import *

#-------------------------------------------------
# Stage2 Emulator and Emulator DQM Sequences

# sequence to run for every event
l1tStage2EmulatorOnlineDQM = cms.Sequence(
    l1tdeStage2Bmtf +
    l1tdeStage2BmtfSecond +
    l1tdeStage2Omtf +
    l1tdeCSCTPG +
    l1tdeStage2CppfOnlineDQMSeq +
    l1tdeStage2EmtfOnlineDQMSeq +
    l1tStage2uGMTEmulatorOnlineDQMSeq +
    l1tdeStage2uGT +
    l1tStage2uGtEmul
)

_run3_l1tStage2EmulatorOnlineDQM = l1tStage2EmulatorOnlineDQM.copy()
_run3_l1tStage2EmulatorOnlineDQM += l1tdeGEMTPG

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
run3_GEM.toReplaceWith( l1tStage2EmulatorOnlineDQM, _run3_l1tStage2EmulatorOnlineDQM )

_run3shower_l1tStage2EmulatorOnlineDQM = l1tStage2EmulatorOnlineDQM.copy()
run3_GEM.toReplaceWith( l1tStage2EmulatorOnlineDQM, cms.Sequence(_run3shower_l1tStage2EmulatorOnlineDQM + l1tdeCSCTPGShower) )

# sequence to run only for validation events
l1tStage2EmulatorOnlineDQMValidationEvents = cms.Sequence(
    l1tdeStage2CaloLayer1 +
    l1tdeStage2CaloLayer2 +
    l1tStage2CaloLayer2Emul
)
