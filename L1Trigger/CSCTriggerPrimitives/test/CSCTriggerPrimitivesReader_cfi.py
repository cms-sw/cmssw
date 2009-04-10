import FWCore.ParameterSet.Config as cms

# Default parameters for CSCTriggerPrimitives analyzer
# =====================================================
lctreader = cms.EDFilter("CSCTriggerPrimitivesReader",
    # Switch on/off the verbosity and turn on/off histogram production
    debug = cms.untracked.bool(False),
    # Define which LCTs are present in the input file.  This will determine the
    # workflow of the Reader.
    dataLctsIn = cms.bool(True),
    emulLctsIn = cms.bool(True),
    # Flag to indicate MTCC data (used only when dataLctsIn = true).
    isMTCCData = cms.bool(False),
    # Labels to retrieve LCTs from the event (optional)
    #                                       produced by unpacker
    CSCLCTProducerData = cms.untracked.string("cscunpacker"),
    #                                       produced by emulator
    CSCLCTProducerEmul = cms.untracked.string("cscTriggerPrimitiveDigis"),
    # Labels to retrieve comparator and wire digis.
    #  (Used only when emulLctsIn = true.)
    CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
)



