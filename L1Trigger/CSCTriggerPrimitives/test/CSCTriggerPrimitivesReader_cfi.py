import FWCore.ParameterSet.Config as cms

# Default parameters for CSCTriggerPrimitives analyzer
# =====================================================
lctreader = cms.EDAnalyzer("CSCTriggerPrimitivesReader",
    # Switch on/off the verbosity and turn on/off histogram production
    debug = cms.untracked.bool(False),
    # Define which LCTs are present in the input file.  This will determine the
    # workflow of the Reader.
    dataLctsIn = cms.bool(True),
    emulLctsIn = cms.bool(True),
    printps = cms.bool(True),
    # Flag to indicate MTCC data (used only when dataLctsIn = true).
    isMTCCData = cms.bool(False),
    # Labels to retrieve LCTs from the event (optional)
    #                                       produced by unpacker
    CSCLCTProducerData = cms.untracked.string("muonCSCDigis"),
    #                                       produced by emulator
    CSCLCTProducerEmul = cms.untracked.string("cscTriggerPrimitiveDigis"),
    # Labels to retrieve simHits, comparator and wire digis.
    #  (Used only when emulLctsIn = true.)
    CSCSimHitProducer = cms.InputTag("g4SimHits", "MuonCSCHits"),  # Full sim.
    #CSCSimHitProducer = cms.InputTag("MuonSimHits", "MuonCSCHits"), # Fast sim.
    CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    checkBadChambers = cms.untracked.bool(True),
    dataIsAnotherMC = cms.untracked.bool(False)
)
