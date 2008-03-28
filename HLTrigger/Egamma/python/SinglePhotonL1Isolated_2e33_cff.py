# The following comments couldn't be translated into the new config version:

#doL1T &

#		           doLocalEcal &

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1IsoSinglePhotonL1MatchFilter = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltEgammaEtFilter_cfi import *
#--------------------------------------------------------------------#
# ET CUT
#--------------------------------------------------------------------#
hltL1IsoSinglePhotonEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltPhotonEcalIsolFilter_cfi import *
#      int32  ncandcut = 1
#--------------------------------------------------------------------#
# ECAL ISOL
#--------------------------------------------------------------------#
hltL1IsoSinglePhotonEcalIsolFilter = copy.deepcopy(hltPhotonEcalIsolFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#double ecalisolcut = 1.5
#int32 ncandcut = 1
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1IsoSinglePhotonHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltPhotonTrackIsolFilter_cfi import *
#      int32 ncandcut = 1
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1IsoSinglePhotonTrackIsolFilter = copy.deepcopy(hltPhotonTrackIsolFilter)
#      double numtrackisolcut = 1
#      int32 ncandcut = 1
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
singlePhotonL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

singlePhotonL1Isolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedSingle")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+hltL1IsoSinglePhotonL1MatchFilter+hltL1IsoSinglePhotonEtFilter+cms.SequencePlaceholder("l1IsolatedPhotonEcalIsol")+hltL1IsoSinglePhotonEcalIsolFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedPhotonHcalIsol")+hltL1IsoSinglePhotonHcalIsolFilter+cms.SequencePlaceholder("doLocalTracker")+cms.SequencePlaceholder("l1IsoEgammaRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoPhotonTrackIsol")+hltL1IsoSinglePhotonTrackIsolFilter+singlePhotonL1IsoPresc)
hltL1IsoSinglePhotonL1MatchFilter.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1IsoSinglePhotonL1MatchFilter.L1SeedFilterTag = 'l1seedSingle'
hltL1IsoSinglePhotonEtFilter.inputTag = 'hltL1IsoSinglePhotonL1MatchFilter'
hltL1IsoSinglePhotonEtFilter.etcut = 80.0
hltL1IsoSinglePhotonEcalIsolFilter.candTag = 'hltL1IsoSinglePhotonEtFilter'
hltL1IsoSinglePhotonEcalIsolFilter.isoTag = 'l1IsolatedPhotonEcalIsol'
hltL1IsoSinglePhotonHcalIsolFilter.candTag = 'hltL1IsoSinglePhotonEcalIsolFilter'
hltL1IsoSinglePhotonHcalIsolFilter.isoTag = 'l1IsolatedPhotonHcalIsol'
hltL1IsoSinglePhotonHcalIsolFilter.hcalisolbarrelcut = 6.
hltL1IsoSinglePhotonHcalIsolFilter.hcalisolendcapcut = 4.
hltL1IsoSinglePhotonTrackIsolFilter.candTag = 'hltL1IsoSinglePhotonHcalIsolFilter'
hltL1IsoSinglePhotonTrackIsolFilter.isoTag = 'l1IsoPhotonTrackIsol'

