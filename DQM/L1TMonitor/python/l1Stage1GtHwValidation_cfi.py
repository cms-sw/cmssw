import FWCore.ParameterSet.Config as cms

# L1GtHwValidation DQM 
#     
# V.M. Ghete 2009-10-09

l1Stage1GtHwValidation = cms.EDAnalyzer("L1GtHwValidation",

    # input tag for the L1 GT hardware DAQ record
    L1GtDataDaqInputTag = cms.InputTag("gtDigis"),
    
    # input tag for the L1 GT hardware EVM record
    L1GtDataEvmInputTag = cms.InputTag("gtEvmDigis"),
    
    # input tag for the L1 GT emulator DAQ record
    L1GtEmulDaqInputTag = cms.InputTag("valStage1GtDigis"),
    
    # input tag for the L1 GT emulator EVM record
    L1GtEmulEvmInputTag = cms.InputTag("valStage1GtDigis"),
    
    # input tag for the L1 GCT hardware record 
    L1GctDataInputTag = cms.InputTag("gctDigis"),   

    DirName = cms.untracked.string("L1TEMU/GTexpert"), 
    
    # exclude algorithm triggers from comparison data - emulator by 
    # condition category and / or type and / or L1 GT object 
    # see CondFormats/L1TObjects/interface/L1GtFwd.h
    #     enum L1GtConditionCategory
    #     enum L1GtConditionType 
    # and DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h
    #     enum L1GtObject
    # 
    # if category given and type empty and object empty, exclude all triggers 
    #     containing conditions of that category
    # if category given and type given and object empty, exclude all triggers 
    #     containing conditions of that category and that type
    # if category given and type given and object given, exclude all triggers 
    #     containing conditions of that category and of that type and that objects 
    # if category empty and type given and object given, exclude all triggers 
    #     of that type and that objects (all categories)
    # ... and so on 
    
    ExcludeCondCategTypeObject = cms.VPSet(
                                        cms.PSet(
                                                 ExcludedCondCategory = cms.string(""),
                                                 ExcludedCondType = cms.string(""),
                                                 ExcludedL1GtObject  = cms.string("GtExternal")
                                                 )
                                        ),
    # exclude algorithm triggers from comparison data - emulator by algorithm name
    # if the corresponding algorithm trigger is not in the menu, nothing will happen
    ExcludeAlgoTrigByName = cms.vstring(),
    #
    
    # exclude algorithm triggers from comparison data - emulator by algorithm bit number
    ExcludeAlgoTrigByBit = cms.vint32()
    #    
)
