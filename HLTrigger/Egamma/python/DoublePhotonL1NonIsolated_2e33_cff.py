# The following comments couldn't be translated into the new config version:

#                           & doLocalEcal

# & hltL1IsoDoublePhotonL1MatchFilter

#
# UP TO HERE OK

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1NonIsoDoublePhotonL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltEgammaEtFilter_cfi import *
#--------------------------------------------------------------------#
# ET CUT
#--------------------------------------------------------------------#
hltL1NonIsoDoublePhotonEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltPhotonEcalIsolFilter_cfi import *
#module hltL1NonIsoDoublePhotonEtFilter = hltEgammaDoubleEtFilter from "HLTrigger/Egamma/data/hltEgammaDoubleEtFilter.cfi"
#replace hltL1NonIsoDoublePhotonEtFilter.candTag =  hltL1NonIsoDoublePhotonL1MatchFilterRegional
#      double etcut1 = 30.0
#      double etcut2 = 20.0
#      int32  npaircut = 1
#--------------------------------------------------------------------#
# ECAL ISOL
#--------------------------------------------------------------------#
hltL1NonIsoDoublePhotonEcalIsolFilter = copy.deepcopy(hltPhotonEcalIsolFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1NonIsoDoublePhotonHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltPhotonTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1NonIsoDoublePhotonTrackIsolFilter = copy.deepcopy(hltPhotonTrackIsolFilter)
import copy
from HLTrigger.Egamma.hltEgammaDoubleEtFilter_cfi import *
#--------------------------------------------------------------------#
# DOUBLE ET CUT
#--------------------------------------------------------------------#
hltL1NonIsoDoublePhotonDoubleEtFilter = copy.deepcopy(hltEgammaDoubleEtFilter)
#      double etcut1 = 30.0
#      double etcut2 = 20.0
#      int32  npaircut = 1
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
doublePhotonL1NonIsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

doublePhotonL1NonIsolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedRelaxedDouble")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1NonIsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+cms.SequencePlaceholder("l1NonIsoRecoEcalCandidate")+hltL1NonIsoDoublePhotonL1MatchFilterRegional+hltL1NonIsoDoublePhotonEtFilter+cms.SequencePlaceholder("l1IsolatedPhotonEcalIsol")+cms.SequencePlaceholder("l1NonIsolatedPhotonEcalIsol")+hltL1NonIsoDoublePhotonEcalIsolFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedPhotonHcalIsol")+cms.SequencePlaceholder("l1NonIsolatedPhotonHcalIsol")+hltL1NonIsoDoublePhotonHcalIsolFilter+cms.SequencePlaceholder("doLocalTracker")+cms.SequencePlaceholder("l1IsoEgammaRegionalRecoTracker")+cms.SequencePlaceholder("l1NonIsoEgammaRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoPhotonTrackIsol")+cms.SequencePlaceholder("l1NonIsoPhotonTrackIsol")+hltL1NonIsoDoublePhotonTrackIsolFilter+hltL1NonIsoDoublePhotonDoubleEtFilter+doublePhotonL1NonIsoPresc)
hltL1NonIsoDoublePhotonL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1NonIsoDoublePhotonL1MatchFilterRegional.candNonIsolatedTag = 'l1NonIsoRecoEcalCandidate'
hltL1NonIsoDoublePhotonL1MatchFilterRegional.L1SeedFilterTag = 'l1seedRelaxedDouble'
hltL1NonIsoDoublePhotonL1MatchFilterRegional.ncandcut = 2
hltL1NonIsoDoublePhotonL1MatchFilterRegional.doIsolated = False
#replace hltL1NonIsoDoublePhotonEtFilter.inputTag = hltL1NonIsoDoublePhotonL1MatchFilter
hltL1NonIsoDoublePhotonEtFilter.inputTag = 'hltL1NonIsoDoublePhotonL1MatchFilterRegional'
hltL1NonIsoDoublePhotonEtFilter.etcut = 20.0
hltL1NonIsoDoublePhotonEtFilter.ncandcut = 2
hltL1NonIsoDoublePhotonEcalIsolFilter.candTag = 'hltL1NonIsoDoublePhotonEtFilter'
hltL1NonIsoDoublePhotonEcalIsolFilter.isoTag = 'l1IsolatedPhotonEcalIsol'
hltL1NonIsoDoublePhotonEcalIsolFilter.nonIsoTag = 'l1NonIsolatedPhotonEcalIsol'
hltL1NonIsoDoublePhotonEcalIsolFilter.ecalisolcut = 2.5
hltL1NonIsoDoublePhotonEcalIsolFilter.ncandcut = 2
hltL1NonIsoDoublePhotonEcalIsolFilter.doIsolated = False
hltL1NonIsoDoublePhotonHcalIsolFilter.candTag = 'hltL1NonIsoDoublePhotonEcalIsolFilter'
hltL1NonIsoDoublePhotonHcalIsolFilter.isoTag = 'l1IsolatedPhotonHcalIsol'
hltL1NonIsoDoublePhotonHcalIsolFilter.nonIsoTag = 'l1NonIsolatedPhotonHcalIsol'
hltL1NonIsoDoublePhotonHcalIsolFilter.hcalisolbarrelcut = 8.
hltL1NonIsoDoublePhotonHcalIsolFilter.hcalisolendcapcut = 6.
hltL1NonIsoDoublePhotonHcalIsolFilter.ncandcut = 2
hltL1NonIsoDoublePhotonHcalIsolFilter.doIsolated = False
hltL1NonIsoDoublePhotonTrackIsolFilter.candTag = 'hltL1NonIsoDoublePhotonHcalIsolFilter'
hltL1NonIsoDoublePhotonTrackIsolFilter.isoTag = 'l1IsoPhotonTrackIsol'
hltL1NonIsoDoublePhotonTrackIsolFilter.nonIsoTag = 'l1NonIsoPhotonTrackIsol'
hltL1NonIsoDoublePhotonTrackIsolFilter.numtrackisolcut = 3
hltL1NonIsoDoublePhotonTrackIsolFilter.ncandcut = 2
hltL1NonIsoDoublePhotonTrackIsolFilter.doIsolated = False
hltL1NonIsoDoublePhotonDoubleEtFilter.candTag = 'hltL1NonIsoDoublePhotonTrackIsolFilter'

