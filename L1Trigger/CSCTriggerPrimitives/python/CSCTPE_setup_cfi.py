import FWCore.ParameterSet.Config as cms

from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigis_cfi import *

cscTriggerPrimitiveDigis.CSCComparatorDigiProducer = "muonCSCDigis:MuonCSCComparatorDigi"
cscTriggerPrimitiveDigis.CSCWireDigiProducer = "muonCSCDigis:MuonCSCWireDigi"
cscTriggerPrimitiveDigis.tmbParam.mpcBlockMe1a = 0
cscTriggerPrimitiveDigis.alctParam07.verbosity = 2
cscTriggerPrimitiveDigis.clctParam07.verbosity = 2
cscTriggerPrimitiveDigis.tmbParam.verbosity = 2
cscTriggerPrimitiveDigis.checkBadChambers = cms.untracked.bool(True)

from L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesConfig_cff import *
l1csctpconf.alctParamMTCC2.alctNplanesHitPretrig = 3
l1csctpconf.alctParamMTCC2.alctNplanesHitAccelPretrig = 3
l1csctpconf.clctParam.clctNplanesHitPretrig = 3
l1csctpconf.clctParam.clctHitPersist = 4

lctreader = cms.EDAnalyzer("CSCTriggerPrimitivesDQM",
                           # Switch on/off the verbosity and turn on/off histogram production
                           debug = cms.untracked.bool(True),
                           # Define which LCTs are present in the input file.  This will determine the
                           # workflow of the Reader.
                           dataLctsIn = cms.bool(True),
                           emulLctsIn = cms.bool(True),
                           printps = cms.bool(False),
                           # Flag to indicate MTCC data (used only when dataLctsIn = true).
                           isMTCCData = cms.bool(False),
                           # Labels to retrieve LCTs from the event (optional)
                           #                                       produced by unpacker
                           CSCLCTProducerData = cms.untracked.string("muonCSCDigis"),
                           #    CSCLCTProducerData = cms.untracked.string("cscunpacker"),                         
                           #                                       produced by emulator
                           CSCLCTProducerEmul = cms.untracked.string("cscTriggerPrimitiveDigis"),
                           # Labels to retrieve simHits, comparator and wire digis.
                           #  (Used only when emulLctsIn = true.)
                           CSCSimHitProducer = cms.InputTag("g4SimHits", "MuonCSCHits"),  # Full sim.
                           #CSCSimHitProducer = cms.InputTag("MuonSimHits", "MuonCSCHits"), # Fast sim.
                           CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
                           CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
                           bad_chambers  = cms.untracked.vstring("ME+1/2/15","ME+1/1/20","ME-1/1/34","ME-1/1/15","ME+1/2/36",
                                                                 "ME+1/1/02","ME-1/1/30","ME+1/1/29","ME+1/1/03"),
                           bad_wires = cms.untracked.vstring("ME-1/1/4","ME-1/1/12","ME-1/1/36","ME-1/2/10","ME-2/2/11","ME-3/2/9"),
                           bad_strips = cms.untracked.vstring("ME+1/1/20","ME-1/1/34","ME-3/2/24")
                           )
