import FWCore.ParameterSet.Config as cms

rawPCCProd = cms.EDProducer("RawPCCProducer",
    RawPCCProducerParameters = cms.PSet(
        #Mod factor to count lumi and the string to specify output
        inputPccLabel = cms.string("alcaPCCProducerRandom"),
        ProdInst = cms.string("alcaPCCRandom"),
        resetEveryNLumi = cms.untracked.int32(1),
        outputProductName = cms.untracked.string("rawPCCRandom"),
        modVeto=cms.vint32()
    )

)
