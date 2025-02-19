import FWCore.ParameterSet.Config as cms

l1GtPackUnpackAnalyzer = cms.EDAnalyzer("L1GtPackUnpackAnalyzer",
    
    # input tag for the initial GT DAQ record: 
    #     GT emulator:  gtDigis  
    #     GT unpacker:  l1GtUnpack  
    InitialDaqGtInputTag = cms.InputTag("gtDigis"),

    # input tag for the initial GMT readout collection: 
    #     gmtDigis       = GMT emulator, 
    #     l1GtUnpack     = GT unpacker 
    InitialMuGmtInputTag = cms.InputTag("gmtDigis"),

    # input tag for the final GT DAQ and GMT records: 
    #     GT unpacker:  gtPackedUnpack (cloned unpacker from L1GtPackUnpackAnalyzer.cfg)
    FinalGtGmtInputTag = cms.InputTag("gtPackedUnpack")
    
)


