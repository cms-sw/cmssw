import FWCore.ParameterSet.Config as cms

#
#  Author: N. Marinelli, U. of Notre Dame, US
#
pfConversions = cms.EDProducer("PFConversionsProducer",
    conversionCollection = cms.string(''),
    # outputs
    PFConversionCollection = cms.string(''),
    # inputs
    conversionProducer = cms.string('conversions'),
    #
    debug = cms.bool(False),
    PFRecTracksFromConversions = cms.string('pfRecTracksFromConversions')
)


