import FWCore.ParameterSet.Config as cms

from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi import cscTriggerPrimitiveDigis

# Default parameters for CSCTriggerPrimitives analyzer
# =====================================================
lctreader = cms.EDAnalyzer("CSCTriggerPrimitivesReader",
    # Parameters common for all boards
    commonParam = cscTriggerPrimitiveDigis.commonParam
    # Switch on/off the verbosity and turn on/off histogram production
    debug = cms.untracked.bool(False),
    # Define which LCTs are present in the input file.  This will determine the
    # workflow of the Reader.
    dataLctsIn = cms.bool(True),
    emulLctsIn = cms.bool(True),
    printps = cms.bool(True),
    # Labels to retrieve LCTs from the event (optional)
    #                                       produced by unpacker
    ##   * 'simCscTriggerPrimitiveDigis','MPCSORTED' : simulated trigger primitives (LCTs) from re-emulating CSC digis
    ##   * 'emtfStage2Digis' : real trigger primitives as received by EMTF, unpacked in EventFilter/L1TRawToDigi/
    #data: muonCSCDigis, emtfStage2Digis
    #simulation(emulator): simCscTriggerPrimitiveDig, simEmtfDigis
    #CSCLCTProducerData = cms.untracked.string("simMuonCSCDigis"),
    CSCLCTProducerData = cms.untracked.string("muonCSCDigis"),
    CSCMPCLCTProducerData = cms.untracked.string("emtfStage2Digis"),
    #                                       produced by emulator
    CSCLCTProducerEmul = cms.untracked.string("cscTriggerPrimitiveDigis"),
    #CSCLCTProducerEmul = cms.untracked.string("simCscTriggerPrimitiveDigis"),
    # Labels to retrieve simHits, comparator and wire digis.
    #  (Used only when emulLctsIn = true.)
    CSCSimHitProducer = cms.InputTag("g4SimHits", "MuonCSCHits"),  # Full sim.
    #CSCSimHitProducer = cms.InputTag("MuonSimHits", "MuonCSCHits"), # Fast sim.
    #simulation
    #CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    #CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    #data
    CSCComparatorDigiProducer = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi"),
    CSCWireDigiProducer = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    checkBadChambers = cms.untracked.bool(True),
    dataIsAnotherMC = cms.untracked.bool(False)
)
