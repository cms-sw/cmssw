#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms

# cfi uGT emulator

simGtStage2Digis = cms.EDProducer("L1TGlobalProducer",
    MuonInputTag = cms.InputTag("simGmtStage2Digis"),
    ExtInputTag = cms.InputTag("simGtExtFakeStage2Digis"),
    EGammaInputTag = cms.InputTag("simCaloStage2Digis"),
    TauInputTag = cms.InputTag("simCaloStage2Digis"),
    JetInputTag = cms.InputTag("simCaloStage2Digis"),
    EtSumInputTag = cms.InputTag("simCaloStage2Digis"),
    AlgorithmTriggersUnmasked = cms.bool(False),    
    AlgorithmTriggersUnprescaled = cms.bool(False),
    # deprecated in Mike's version of producer:                              
    #ProduceL1GtDaqRecord = cms.bool(True),
    #GmtInputTag = cms.InputTag("gtInput"),
    #extInputTag = cms.InputTag("gtInput"),
    #caloInputTag = cms.InputTag("gtInput"),
    #AlternativeNrBxBoardDaq = cms.uint32(0),
    #WritePsbL1GtDaqRecord = cms.bool(True),
    #TriggerMenuLuminosity = cms.string('startup'),
    #PrescaleCSVFile = cms.string('prescale_L1TGlobal.csv'),
    #PrescaleSet = cms.uint32(1),
    #BstLengthBytes = cms.int32(-1),
    #Verbosity = cms.untracked.int32(0)
)

