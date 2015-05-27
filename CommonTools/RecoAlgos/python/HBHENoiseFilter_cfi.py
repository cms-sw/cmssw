#
# This is a replacement for the original HBHENoiseFilter configuration.
# See https://twiki.cern.ch/twiki/bin/viewauth/CMS/HCALNoiseFilterRecipe.
# Note that this replacement relies on having the HcalNoiseSummary in the
# event record but not necessarily HBHENoiseFilterResult.
#
import FWCore.ParameterSet.Config as cms

# Module which will remake HBHENoiseFilterResult
from CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi import HBHENoiseFilterResultProducer
MakeHBHENoiseFilterResult = HBHENoiseFilterResultProducer.clone()

# Filter on the standard HCAL noise decision
HBHENoiseFilter = cms.EDFilter(
    'BooleanFlagFilter',
    inputLabel = cms.InputTag('MakeHBHENoiseFilterResult','HBHENoiseFilterResult'),
    reverseDecision = cms.bool(False)
)

# Customize MakeHBHENoiseFilterResult in the same manner
# as HBHENoiseFilterResultProducer
from Configuration.StandardSequences.Eras import eras
eras.run2_common.toModify(MakeHBHENoiseFilterResult, IgnoreTS4TS5ifJetInLowBVRegion=False)
eras.run2_25ns_specific.toModify(MakeHBHENoiseFilterResult, defaultDecision="HBHENoiseFilterResultRun2Loose")
