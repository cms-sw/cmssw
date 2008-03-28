# The following comments couldn't be translated into the new config version:

#                           l1seedDouble &

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1IsoDoubleExclPhotonL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltDoubleExclusivePhotonFilter_cfi import *
#--------------------------------------------------------------------#
# ET AND PHI CUT
#--------------------------------------------------------------------#
hltL1IsoDoubleExclPhotonEtPhiFilter = copy.deepcopy(hltDoubleExclusivePhotonFilter)
import copy
from HLTrigger.Egamma.hltPhotonEcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# ECAL ISOL
#--------------------------------------------------------------------#
hltL1IsoDoubleExclPhotonEcalIsolFilter = copy.deepcopy(hltPhotonEcalIsolFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1IsoDoubleExclPhotonHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltPhotonTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1IsoDoubleExclPhotonTrackIsolFilter = copy.deepcopy(hltPhotonTrackIsolFilter)
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
doubleExclPhotonL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

doubleExclPhotonL1Isolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedExclusiveDouble")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+hltL1IsoDoubleExclPhotonL1MatchFilterRegional+hltL1IsoDoubleExclPhotonEtPhiFilter+cms.SequencePlaceholder("l1IsolatedPhotonEcalIsol")+hltL1IsoDoubleExclPhotonEcalIsolFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedPhotonHcalIsol")+hltL1IsoDoubleExclPhotonHcalIsolFilter+cms.SequencePlaceholder("doLocalTracker")+cms.SequencePlaceholder("l1IsoEgammaRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoPhotonTrackIsol")+hltL1IsoDoubleExclPhotonTrackIsolFilter+doubleExclPhotonL1IsoPresc)
hltL1IsoDoubleExclPhotonL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1IsoDoubleExclPhotonL1MatchFilterRegional.L1SeedFilterTag = 'l1seedExclusiveDouble'
hltL1IsoDoubleExclPhotonL1MatchFilterRegional.ncandcut = 2
hltL1IsoDoubleExclPhotonL1MatchFilterRegional.doIsolated = True
hltL1IsoDoubleExclPhotonEtPhiFilter.candTag = 'hltL1IsoDoubleExclPhotonL1MatchFilterRegional'
hltL1IsoDoubleExclPhotonEcalIsolFilter.candTag = 'hltL1IsoDoubleExclPhotonEtPhiFilter'
hltL1IsoDoubleExclPhotonEcalIsolFilter.isoTag = 'l1IsolatedPhotonEcalIsol'
hltL1IsoDoubleExclPhotonEcalIsolFilter.ecalisolcut = 2.5
hltL1IsoDoubleExclPhotonEcalIsolFilter.ncandcut = 2
hltL1IsoDoubleExclPhotonHcalIsolFilter.candTag = 'hltL1IsoDoubleExclPhotonEcalIsolFilter'
hltL1IsoDoubleExclPhotonHcalIsolFilter.isoTag = 'l1IsolatedPhotonHcalIsol'
hltL1IsoDoubleExclPhotonHcalIsolFilter.hcalisolbarrelcut = 8.
hltL1IsoDoubleExclPhotonHcalIsolFilter.hcalisolendcapcut = 6.
hltL1IsoDoubleExclPhotonHcalIsolFilter.ncandcut = 2
hltL1IsoDoubleExclPhotonTrackIsolFilter.candTag = 'hltL1IsoDoubleExclPhotonHcalIsolFilter'
hltL1IsoDoubleExclPhotonTrackIsolFilter.isoTag = 'l1IsoPhotonTrackIsol'
hltL1IsoDoubleExclPhotonTrackIsolFilter.numtrackisolcut = 3
hltL1IsoDoubleExclPhotonTrackIsolFilter.ncandcut = 2

