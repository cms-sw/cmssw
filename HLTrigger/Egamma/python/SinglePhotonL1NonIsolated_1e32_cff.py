# The following comments couldn't be translated into the new config version:

#doL1T &

# 		           doLocalEcal &

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1NonIsoSinglePhotonL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltEgammaEtFilter_cfi import *
#--------------------------------------------------------------------#
# ET CUT
#--------------------------------------------------------------------#
hltL1NonIsoSinglePhotonEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltPhotonEcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# ECAL ISOL
#--------------------------------------------------------------------#
hltL1NonIsoSinglePhotonEcalIsolFilter = copy.deepcopy(hltPhotonEcalIsolFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1NonIsoSinglePhotonHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltPhotonTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1NonIsoSinglePhotonTrackIsolFilter = copy.deepcopy(hltPhotonTrackIsolFilter)
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
singlePhotonL1NonIsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

singlePhotonL1NonIsolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedRelaxedSingle")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1NonIsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+cms.SequencePlaceholder("l1NonIsoRecoEcalCandidate")+hltL1NonIsoSinglePhotonL1MatchFilterRegional+hltL1NonIsoSinglePhotonEtFilter+cms.SequencePlaceholder("l1IsolatedPhotonEcalIsol")+cms.SequencePlaceholder("l1NonIsolatedPhotonEcalIsol")+hltL1NonIsoSinglePhotonEcalIsolFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedPhotonHcalIsol")+cms.SequencePlaceholder("l1NonIsolatedPhotonHcalIsol")+hltL1NonIsoSinglePhotonHcalIsolFilter+cms.SequencePlaceholder("doLocalTracker")+cms.SequencePlaceholder("l1IsoEgammaRegionalRecoTracker")+cms.SequencePlaceholder("l1NonIsoEgammaRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoPhotonTrackIsol")+cms.SequencePlaceholder("l1NonIsoPhotonTrackIsol")+hltL1NonIsoSinglePhotonTrackIsolFilter+singlePhotonL1NonIsoPresc)
hltL1NonIsoSinglePhotonL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1NonIsoSinglePhotonL1MatchFilterRegional.candNonIsolatedTag = 'l1NonIsoRecoEcalCandidate'
hltL1NonIsoSinglePhotonL1MatchFilterRegional.L1SeedFilterTag = 'l1seedRelaxedSingle'
hltL1NonIsoSinglePhotonL1MatchFilterRegional.ncandcut = 1
hltL1NonIsoSinglePhotonL1MatchFilterRegional.doIsolated = False
hltL1NonIsoSinglePhotonEtFilter.inputTag = 'hltL1NonIsoSinglePhotonL1MatchFilterRegional'
hltL1NonIsoSinglePhotonEtFilter.etcut = 40.0
hltL1NonIsoSinglePhotonEtFilter.ncandcut = 1
hltL1NonIsoSinglePhotonEcalIsolFilter.candTag = 'hltL1NonIsoSinglePhotonEtFilter'
hltL1NonIsoSinglePhotonEcalIsolFilter.isoTag = 'l1IsolatedPhotonEcalIsol'
hltL1NonIsoSinglePhotonEcalIsolFilter.nonIsoTag = 'l1NonIsolatedPhotonEcalIsol'
hltL1NonIsoSinglePhotonEcalIsolFilter.ecalisolcut = 1.5
hltL1NonIsoSinglePhotonEcalIsolFilter.ncandcut = 1
hltL1NonIsoSinglePhotonEcalIsolFilter.doIsolated = False
hltL1NonIsoSinglePhotonHcalIsolFilter.candTag = 'hltL1NonIsoSinglePhotonEcalIsolFilter'
hltL1NonIsoSinglePhotonHcalIsolFilter.isoTag = 'l1IsolatedPhotonHcalIsol'
hltL1NonIsoSinglePhotonHcalIsolFilter.nonIsoTag = 'l1NonIsolatedPhotonHcalIsol'
hltL1NonIsoSinglePhotonHcalIsolFilter.hcalisolbarrelcut = 6.
hltL1NonIsoSinglePhotonHcalIsolFilter.hcalisolendcapcut = 4.
hltL1NonIsoSinglePhotonHcalIsolFilter.ncandcut = 1
hltL1NonIsoSinglePhotonHcalIsolFilter.doIsolated = False
hltL1NonIsoSinglePhotonTrackIsolFilter.candTag = 'hltL1NonIsoSinglePhotonHcalIsolFilter'
hltL1NonIsoSinglePhotonTrackIsolFilter.isoTag = 'l1IsoPhotonTrackIsol'
hltL1NonIsoSinglePhotonTrackIsolFilter.nonIsoTag = 'l1NonIsoPhotonTrackIsol'
hltL1NonIsoSinglePhotonTrackIsolFilter.numtrackisolcut = 1
hltL1NonIsoSinglePhotonTrackIsolFilter.ncandcut = 1
hltL1NonIsoSinglePhotonTrackIsolFilter.doIsolated = False

