# The following comments couldn't be translated into the new config version:

#				& doLocalEcal

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltEgammaEtFilter_cfi import *
#--------------------------------------------------------------------#
# ET CUT
#--------------------------------------------------------------------#
hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter = copy.deepcopy(hltEgammaEtFilter)
#--------------------------------------------------------------------#
# Prescaler
#--------------------------------------------------------------------#
singleEMVHEL1NonIsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

singleEMVHEL1NonIsolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedRelaxedSingle")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1NonIsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+cms.SequencePlaceholder("l1NonIsoRecoEcalCandidate")+hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional+hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter+singleEMVHEL1NonIsoPresc)
hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional.candNonIsolatedTag = 'l1NonIsoRecoEcalCandidate'
hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional.L1SeedFilterTag = 'l1seedRelaxedSingle'
hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional.ncandcut = 1
hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional.doIsolated = False
hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter.inputTag = 'hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional'
hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter.etcut = 200.0
hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter.ncandcut = 1

