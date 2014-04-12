import FWCore.ParameterSet.Config as cms

#
# producer for GEDphotonCore collection
#
gedPhotonCore = cms.EDProducer("GEDPhotonCoreProducer",
 #   conversionProducer = cms.InputTag("conversions"),
    pfEgammaCandidates = cms.InputTag("particleFlowEGamma"),
    pixelSeedProducer = cms.InputTag('electronMergedSeeds'),
    gedPhotonCoreCollection = cms.string('')

)


