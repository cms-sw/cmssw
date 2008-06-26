import FWCore.ParameterSet.Config as cms

l1GtDataEmulAnalyzer = cms.EDAnalyzer("L1GtDataEmulAnalyzer",

    # input tag for the L1 GT hardware DAQ/EVM record
    L1GtDataInputTag = cms.InputTag("gtDigis"),
    #
    # input tag for the L1 GT emulator DAQ/EVM record
    L1GtEmulInputTag = cms.InputTag("l1GtEmulDigis"),
    #
    # input tag for the L1 GCT hardware record 
    L1GctDataInputTag = cms.InputTag("gctDigis")   
)


