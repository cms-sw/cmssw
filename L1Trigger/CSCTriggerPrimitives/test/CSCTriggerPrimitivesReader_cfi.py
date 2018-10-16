import FWCore.ParameterSet.Config as cms

# Default parameters for CSCTriggerPrimitives analyzer
# =====================================================
lctreader = cms.EDAnalyzer("CSCTriggerPrimitivesReader",
    # Parameters common for all boards
    commonParam = cms.PSet(
        isTMB07 = cms.bool(True),
        isMTCC = cms.bool(False),
        
        # Flag for SLHC studies (upgraded ME11, MPC)
        # (if true, isTMB07 should be true as well)
        isSLHC = cms.bool(False),

        # ME1a configuration:
        # smartME1aME1b=f, gangedME1a=t
        #   default logic for current HW
        # smartME1aME1b=t, gangedME1a=f
        #   realistic upgrade scenario: 
        #   one ALCT finder and two CLCT finders per ME11
        #   with additional logic for A/CLCT matching with ME1a unganged
        # smartME1aME1b=t, gangedME1a=t
        #   the previous case with ME1a still being ganged
        # Note: gangedME1a has effect only if smartME1aME1b=t
        smartME1aME1b = cms.bool(False),
        gangedME1a = cms.bool(False),
        
        # flags to optionally disable finding stubs in ME42 or ME1a
        disableME1a = cms.bool(False),
        disableME42 = cms.bool(False),

        ## enable the GEM-CSC integrated triggers for ME11 or ME21
        runME11ILT = cms.bool(False),
        runME21ILT = cms.bool(False),
        runME3141ILT = cms.bool(False),
	),
    # Switch on/off the verbosity and turn on/off histogram production
    debug = cms.untracked.bool(False),
    # Define which LCTs are present in the input file.  This will determine the
    # workflow of the Reader.
    dataLctsIn = cms.bool(True),
    emulLctsIn = cms.bool(True),
    printps = cms.bool(True),
    # Flag to indicate MTCC data (used only when dataLctsIn = true).
    #isMTCCData = cms.bool(False),
    # Labels to retrieve LCTs from the event (optional)
    #                                       produced by unpacker
    ##   * 'simCscTriggerPrimitiveDigis','MPCSORTED' : simulated trigger primitives (LCTs) from re-emulating CSC digis
    ##   * 'csctfDigis' : real trigger primitives as received by CSCTF (legacy trigger)
    ##   * 'emtfStage2Digis' : real trigger primitives as received by EMTF, unpacked in EventFilter/L1TRawToDigi/
    #data: muonCSCDigis, emtfStage2Digis
    #simualtion(emulator): simCscTriggerPrimitiveDig, simEmtfDigis
    #CSCLCTProducerData = cms.untracked.string("simMuonCSCDigis"),
    CSCLCTProducerData = cms.untracked.string("muonCSCDigis"),
    CSCMPCLCTProducerData = cms.untracked.string("csctfDigis"),
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
