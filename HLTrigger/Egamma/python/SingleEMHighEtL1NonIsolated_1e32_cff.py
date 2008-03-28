# The following comments couldn't be translated into the new config version:

#					& doLocalEcal

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1NonIsoSingleEMHighEtL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltEgammaEtFilter_cfi import *
#--------------------------------------------------------------------#
# ET CUT
#--------------------------------------------------------------------#
hltL1NonIsoSinglePhotonEMHighEtEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltPhotonEcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# ECAL ISOL
#--------------------------------------------------------------------#
hltL1NonIsoSingleEMHighEtEcalIsolFilter = copy.deepcopy(hltPhotonEcalIsolFilter)
import copy
from HLTrigger.Egamma.hltSingleEMHOEFilter_cfi import *
#--------------------------------------------------------------------#
# H/E
#--------------------------------------------------------------------#
hltL1NonIsoSingleEMHighEtHOEFilter = copy.deepcopy(hltSinlgeEMHOEFilter)
import copy
from HLTrigger.Egamma.hltSingleEMHcalDoubleConeFilter_cfi import *
#      InputTag isoTag  = l1IsolatedElectronHcalIsol
#      InputTag nonIsoTag  = l1NonIsolatedElectronHcalIsol
#      double hcalisolbarrelcut = 0.05
#      double hcalisolendcapcut = 0.05
#      int32 ncandcut = 1
#      bool doIsolated = false
#--------------------------------------------------------------------#
# HCAL DOUBLE CONE
#--------------------------------------------------------------------#
hltL1NonIsoSingleEMHighEtHcalDBCFilter = copy.deepcopy(hltSingleEMHcalDoubleConeFilter)
import copy
from HLTrigger.Egamma.hltPhotonTrackIsolFilter_cfi import *
#     InputTag candTag = hltL1NonIsoSingleEMHighEtHOEFilter
#     InputTag isoTag  = hltHcalDoubleCone
#     InputTag nonIsoTag  = l1NonIsoEMHcalDoubleCone
#     double hcalisolbarrelcut = 8.
#     double hcalisolendcapcut = 8.
#     int32 ncandcut = 1
#     bool doIsolated = false
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1NonIsoSingleEMHighEtTrackIsolFilter = copy.deepcopy(hltPhotonTrackIsolFilter)
#--------------------------------------------------------------------#
# Prescaler
#--------------------------------------------------------------------#
singleEMVHighEtL1NonIsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

singleEMHighEtL1NonIsolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedRelaxedSingle")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1NonIsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+cms.SequencePlaceholder("l1NonIsoRecoEcalCandidate")+hltL1NonIsoSingleEMHighEtL1MatchFilterRegional+hltL1NonIsoSinglePhotonEMHighEtEtFilter+cms.SequencePlaceholder("l1IsolatedPhotonEcalIsol")+cms.SequencePlaceholder("l1NonIsolatedPhotonEcalIsol")+hltL1NonIsoSingleEMHighEtEcalIsolFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1NonIsolatedElectronHcalIsol")+cms.SequencePlaceholder("l1IsolatedElectronHcalIsol")+hltL1NonIsoSingleEMHighEtHOEFilter+cms.SequencePlaceholder("hltHcalDoubleCone")+cms.SequencePlaceholder("l1NonIsoEMHcalDoubleCone")+hltL1NonIsoSingleEMHighEtHcalDBCFilter+cms.SequencePlaceholder("doLocalTracker")+cms.SequencePlaceholder("l1IsoEgammaRegionalRecoTracker")+cms.SequencePlaceholder("l1NonIsoEgammaRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoPhotonTrackIsol")+cms.SequencePlaceholder("l1NonIsoPhotonTrackIsol")+hltL1NonIsoSingleEMHighEtTrackIsolFilter+singleEMVHighEtL1NonIsoPresc)
hltL1NonIsoSingleEMHighEtL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1NonIsoSingleEMHighEtL1MatchFilterRegional.candNonIsolatedTag = 'l1NonIsoRecoEcalCandidate'
hltL1NonIsoSingleEMHighEtL1MatchFilterRegional.L1SeedFilterTag = 'l1seedRelaxedSingle'
hltL1NonIsoSingleEMHighEtL1MatchFilterRegional.ncandcut = 1
hltL1NonIsoSingleEMHighEtL1MatchFilterRegional.doIsolated = False
hltL1NonIsoSinglePhotonEMHighEtEtFilter.inputTag = 'hltL1NonIsoSingleEMHighEtL1MatchFilterRegional'
hltL1NonIsoSinglePhotonEMHighEtEtFilter.etcut = 80.0
hltL1NonIsoSinglePhotonEMHighEtEtFilter.ncandcut = 1
hltL1NonIsoSingleEMHighEtEcalIsolFilter.candTag = 'hltL1NonIsoSinglePhotonEMHighEtEtFilter'
hltL1NonIsoSingleEMHighEtEcalIsolFilter.isoTag = 'l1IsolatedPhotonEcalIsol'
hltL1NonIsoSingleEMHighEtEcalIsolFilter.nonIsoTag = 'l1NonIsolatedPhotonEcalIsol'
hltL1NonIsoSingleEMHighEtEcalIsolFilter.ecalisolcut = 5.
hltL1NonIsoSingleEMHighEtEcalIsolFilter.ncandcut = 1
hltL1NonIsoSingleEMHighEtEcalIsolFilter.doIsolated = False
hltL1NonIsoSingleEMHighEtHOEFilter.candTag = 'hltL1NonIsoSingleEMHighEtEcalIsolFilter'
hltL1NonIsoSingleEMHighEtTrackIsolFilter.candTag = 'hltL1NonIsoSingleEMHighEtHcalDBCFilter'
hltL1NonIsoSingleEMHighEtTrackIsolFilter.isoTag = 'l1IsoPhotonTrackIsol'
hltL1NonIsoSingleEMHighEtTrackIsolFilter.nonIsoTag = 'l1NonIsoPhotonTrackIsol'
hltL1NonIsoSingleEMHighEtTrackIsolFilter.numtrackisolcut = 4
hltL1NonIsoSingleEMHighEtTrackIsolFilter.ncandcut = 1
hltL1NonIsoSingleEMHighEtTrackIsolFilter.doIsolated = False

