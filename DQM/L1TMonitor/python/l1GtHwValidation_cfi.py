import FWCore.ParameterSet.Config as cms

# L1GtHwValidation DQM 
#     
# V.M. Ghete 2009-10-09

l1GtHwValidation = cms.EDAnalyzer("L1GtHwValidation",

    # input tag for the L1 GT hardware DAQ record
    L1GtDataDaqInputTag = cms.InputTag("gtDigis"),
    #
    # input tag for the L1 GT hardware EVM record
    L1GtDataEvmInputTag = cms.InputTag("gtEvmDigis"),
    #
    # input tag for the L1 GT emulator DAQ record
    L1GtEmulDaqInputTag = cms.InputTag("valGtDigis"),
    #
    # input tag for the L1 GT emulator EVM record
    L1GtEmulEvmInputTag = cms.InputTag("valGtDigis"),
    #
    # input tag for the L1 GCT hardware record 
    L1GctDataInputTag = cms.InputTag("gctDigis"),   

    DQMStore = cms.untracked.bool(False),
    DirName = cms.untracked.string("L1TEMU/GTHW")
)


