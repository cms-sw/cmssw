import FWCore.ParameterSet.Config as cms

#
# producer for GEDphotonCore collection
#
gedPhotonCore = cms.EDProducer("GEDPhotonCoreProducer",
 #   conversionProducer = cms.InputTag("conversions"),
    pfEgammaCandidates = cms.InputTag("particleFlowEGamma"),
    gedPhotonCoreCollection = cms.string(''),
    pixelSeedProducer = cms.string('electronMergedSeeds')
#    minSCEt = cms.double(10.0)
)


