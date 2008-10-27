import FWCore.ParameterSet.Config as cms

#
#  Author: N. Marinelli, U. of Notre Dame, US
#
pfConversions = cms.EDProducer("PFConversionsProducer",
    conversionCollection = cms.string(''),
    # outputs
    PFConversionCollection = cms.string(''),
    # inputs
    #   string conversionProducer = "conversions"
    conversionProducer = cms.string('photons'),
    #
    debug = cms.bool(True),
    PFRecTracksFromConversions = cms.string('pfRecTracksFromConversions')
)


