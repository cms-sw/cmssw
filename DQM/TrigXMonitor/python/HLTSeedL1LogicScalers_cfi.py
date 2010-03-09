import FWCore.ParameterSet.Config as cms

hltSeedL1Logic = cms.EDAnalyzer('HLTSeedL1LogicScalers',

    DQMFolder = cms.untracked.string("HLT/HLTSeedL1LogicScalers_EvF"),
    l1GtLabel = cms.InputTag("l1GtUnpack","","HLT"),
    processname = cms.untracked.string("HLT"),
    l1BeforeMask = cms.untracked.bool(False),
    monitorPaths = cms.vstring(
            'HLT_L1MuOpen',
            'HLT_L1Mu',
            'HLT_Mu3',
            'HLT_L1SingleForJet',
            'HLT_SingleLooseIsoTau20',
            'HLT_MinBiasEcal'
      ),
    # input tag for GT readout collection: 
    #     GT emulator, GT unpacker:  gtDigis  
    L1GtDaqReadoutRecordInputTag = cms.InputTag("gtDigis"),
    
    # input tags for GT lite record
    #     L1 GT lite record producer:  l1GtRecord  
    L1GtRecordInputTag = cms.InputTag("l1GtRecord")

)
