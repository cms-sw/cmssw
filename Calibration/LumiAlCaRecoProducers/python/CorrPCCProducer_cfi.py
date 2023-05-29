import FWCore.ParameterSet.Config as cms

#Make sure that variables match in producer.cc and .h
corrPCCProd = cms.EDProducer("CorrPCCProducer",
    CorrPCCProducerParameters = cms.PSet(
        #Mod factor to count lumi and the string to specify output
        inLumiObLabel = cms.string("rawPCCProd"),
        ProdInst = cms.string("rawPCCRandom"),
        approxLumiBlockSize=cms.int32(50),
        trigstring = cms.untracked.string("corrPCCRand"),
        type2_a= cms.double(0.00094),
        type2_b= cms.double(0.018),
    )
)
