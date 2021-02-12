import FWCore.ParameterSet.Config as cms

from ..modules.hltDeepBLifetimeTagInfosPFPuppi_cfi import *
from ..modules.hltPfJetProbabilityBJetTagsPuppi_cfi import *
from ..modules.hltPFPuppiJetForBtagEta2p4_cfi import *
from ..modules.hltPFPuppiJetForBtagEta4p0_cfi import *
from ..modules.hltPFPuppiJetForBtagSelectorEta2p4_cfi import *
from ..modules.hltPFPuppiJetForBtagSelectorEta4p0_cfi import *

HLTBtagProbabiltySequencePFPuppi = cms.Sequence(hltPFPuppiJetForBtagSelectorEta2p4+hltPFPuppiJetForBtagSelectorEta4p0+hltPFPuppiJetForBtagEta2p4+hltPFPuppiJetForBtagEta4p0+hltDeepBLifetimeTagInfosPFPuppi+hltPfJetProbabilityBJetTagsPuppi)
