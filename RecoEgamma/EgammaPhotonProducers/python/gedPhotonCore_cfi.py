import FWCore.ParameterSet.Config as cms

#
# producer for GEDphotonCore collection
# $Id: photonCore_cfi.py,v 1.4 2010/05/11 16:14:12 nancy Exp $
#
gedPhotonCore = cms.EDProducer("GEDPhotonCoreProducer",
 #   conversionProducer = cms.InputTag("conversions"),
    pfEgammaCandidates = cms.InputTag("particleFlowEGamma"),
    gedPhotonCoreCollection = cms.string(''),
    pixelSeedProducer = cms.string('electronMergedSeeds')
#    minSCEt = cms.double(10.0)
)


