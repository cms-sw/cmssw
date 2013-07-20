import FWCore.ParameterSet.Config as cms

#
# producer for GEDphotonCore collection
# $Id: gedPhotonCore_cfi.py,v 1.1 2013/05/07 12:35:01 nancy Exp $
#
gedPhotonCore = cms.EDProducer("GEDPhotonCoreProducer",
 #   conversionProducer = cms.InputTag("conversions"),
    pfEgammaCandidates = cms.InputTag("particleFlowEGamma"),
    gedPhotonCoreCollection = cms.string(''),
    pixelSeedProducer = cms.string('electronMergedSeeds')
#    minSCEt = cms.double(10.0)
)


