#
# This is a replacement for the original HBHENoiseFilter configuration.
# See https://twiki.cern.ch/twiki/bin/viewauth/CMS/HCALNoiseFilterRecipe.
# Note that this replacement relies on the HBHENoiseFilterResultProducer
# output.
#
import FWCore.ParameterSet.Config as cms

HBHENoiseFilter = cms.EDFilter(
    'BooleanFlagFilter',
    inputLabel = cms.InputTag('HBHENoiseFilterResultProducer','HBHENoiseFilterResult'),
    reverseDecision = cms.bool(False)
)
