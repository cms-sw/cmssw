# The following comments couldn't be translated into the new config version:

#doL1T &

#                           doLocalEcal &

#hltL1IsoDoublePhotonL1MatchFilter &

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1IsoDoublePhotonL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltEgammaEtFilter_cfi import *
#
#--------------------------------------------------------------------#
# ET CUT
#--------------------------------------------------------------------#
hltL1IsoDoublePhotonEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltPhotonEcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# ECAL ISOL
#--------------------------------------------------------------------#
hltL1IsoDoublePhotonEcalIsolFilter = copy.deepcopy(hltPhotonEcalIsolFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1IsoDoublePhotonHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltPhotonTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1IsoDoublePhotonTrackIsolFilter = copy.deepcopy(hltPhotonTrackIsolFilter)
import copy
from HLTrigger.Egamma.hltEgammaDoubleEtFilter_cfi import *
#--------------------------------------------------------------------#
# DOUBLE ET CUT
#--------------------------------------------------------------------#
hltL1IsoDoublePhotonDoubleEtFilter = copy.deepcopy(hltEgammaDoubleEtFilter)
#      double etcut1 = 30.0
#      double etcut2 = 20.0
#      int32  npaircut = 1
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
doublePhotonL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

doublePhotonL1Isolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedDouble")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+hltL1IsoDoublePhotonL1MatchFilterRegional+hltL1IsoDoublePhotonEtFilter+cms.SequencePlaceholder("l1IsolatedPhotonEcalIsol")+hltL1IsoDoublePhotonEcalIsolFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedPhotonHcalIsol")+hltL1IsoDoublePhotonHcalIsolFilter+cms.SequencePlaceholder("doLocalTracker")+cms.SequencePlaceholder("l1IsoEgammaRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoPhotonTrackIsol")+hltL1IsoDoublePhotonTrackIsolFilter+hltL1IsoDoublePhotonDoubleEtFilter+doublePhotonL1IsoPresc)
hltL1IsoDoublePhotonL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1IsoDoublePhotonL1MatchFilterRegional.L1SeedFilterTag = 'l1seedDouble'
hltL1IsoDoublePhotonL1MatchFilterRegional.ncandcut = 2
hltL1IsoDoublePhotonL1MatchFilterRegional.doIsolated = True
hltL1IsoDoublePhotonEtFilter.inputTag = 'hltL1IsoDoublePhotonL1MatchFilterRegional'
hltL1IsoDoublePhotonEtFilter.etcut = 20.0
hltL1IsoDoublePhotonEtFilter.ncandcut = 2
hltL1IsoDoublePhotonEcalIsolFilter.candTag = 'hltL1IsoDoublePhotonEtFilter'
hltL1IsoDoublePhotonEcalIsolFilter.isoTag = 'l1IsolatedPhotonEcalIsol'
hltL1IsoDoublePhotonEcalIsolFilter.ecalisolcut = 2.5
hltL1IsoDoublePhotonEcalIsolFilter.ncandcut = 2
hltL1IsoDoublePhotonHcalIsolFilter.candTag = 'hltL1IsoDoublePhotonEcalIsolFilter'
hltL1IsoDoublePhotonHcalIsolFilter.isoTag = 'l1IsolatedPhotonHcalIsol'
hltL1IsoDoublePhotonHcalIsolFilter.hcalisolbarrelcut = 8.
hltL1IsoDoublePhotonHcalIsolFilter.hcalisolendcapcut = 6.
hltL1IsoDoublePhotonHcalIsolFilter.ncandcut = 2
hltL1IsoDoublePhotonTrackIsolFilter.candTag = 'hltL1IsoDoublePhotonHcalIsolFilter'
hltL1IsoDoublePhotonTrackIsolFilter.isoTag = 'l1IsoPhotonTrackIsol'
hltL1IsoDoublePhotonTrackIsolFilter.numtrackisolcut = 3
hltL1IsoDoublePhotonTrackIsolFilter.ncandcut = 2
hltL1IsoDoublePhotonDoubleEtFilter.candTag = 'hltL1IsoDoublePhotonTrackIsolFilter'
hltL1IsoDoublePhotonDoubleEtFilter.etcut1 = 20.0

