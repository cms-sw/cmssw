#
# This is a replacement for the original HBHENoiseFilter configuration.
# See https://twiki.cern.ch/twiki/bin/viewauth/CMS/HCALNoiseFilterRecipe.
# This replacement relies on having the HBHENoiseFilterResult in the
# event record.
#
import FWCore.ParameterSet.Config as cms

# Filter on the standard HCAL noise decision
HBHENoiseFilter = cms.EDFilter(
    'BooleanFlagFilter',
    inputLabel = cms.InputTag('HBHENoiseFilterResultProducer','HBHENoiseFilterResult'),
    reverseDecision = cms.bool(False)
)
