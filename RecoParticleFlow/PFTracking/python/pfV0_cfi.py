import FWCore.ParameterSet.Config as cms

pfV0 = cms.EDProducer(
    "PFV0Producer", 
    V0List = cms.VInputTag(cms.InputTag("generalV0Candidates","Kshort"),
                           cms.InputTag("generalV0Candidates","Lambda")
                           )
                  
)


