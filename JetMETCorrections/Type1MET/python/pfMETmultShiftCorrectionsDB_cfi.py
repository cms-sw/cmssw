import FWCore.ParameterSet.Config as cms

# Using DB for miniAOD
pfMEtMultShiftCorrDB = cms.EDProducer("MultShiftMETcorrDBInputProducer",
    srcPFlow = cms.InputTag('packedPFCandidates', ''),
    vertexCollection = cms.InputTag('offlineSlimmedPrimaryVertices'),
    isData = cms.bool(False),
    payloadName = cms.string('PfType1Met'),
    #sampleType = cms.string('MC') # MC, Data, DY, TTJets, WJets: MC is default, Data don't need to be specified because of "isData".
    )

