import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1IsoSinglePhotonPrescaledL1MatchFilter = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltEgammaEtFilter_cfi import *
#--------------------------------------------------------------------#
# ET CUT
#--------------------------------------------------------------------#
hltL1IsoSinglePhotonPrescaledEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltPhotonEcalIsolFilter_cfi import *
#      int32  ncandcut = 1
#--------------------------------------------------------------------#
# ECAL ISOL
#--------------------------------------------------------------------#
hltL1IsoSinglePhotonPrescaledEcalIsolFilter = copy.deepcopy(hltPhotonEcalIsolFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#double ecalisolcut = 1.5
#int32 ncandcut = 1
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1IsoSinglePhotonPrescaledHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltPhotonTrackIsolFilter_cfi import *
#      int32 ncandcut = 1
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1IsoSinglePhotonPrescaledTrackIsolFilter = copy.deepcopy(hltPhotonTrackIsolFilter)
#      double numtrackisolcut = 1
#      int32 ncandcut = 1
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
singlePhotonPrescaledL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

singlePhotonPrescaledL1Isolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedSinglePrescaled")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+hltL1IsoSinglePhotonPrescaledL1MatchFilter+hltL1IsoSinglePhotonPrescaledEtFilter+cms.SequencePlaceholder("l1IsolatedPhotonEcalIsol")+hltL1IsoSinglePhotonPrescaledEcalIsolFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedPhotonHcalIsol")+hltL1IsoSinglePhotonPrescaledHcalIsolFilter+cms.SequencePlaceholder("doLocalTracker")+cms.SequencePlaceholder("l1IsoEgammaRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoPhotonTrackIsol")+hltL1IsoSinglePhotonPrescaledTrackIsolFilter+singlePhotonPrescaledL1IsoPresc)
hltL1IsoSinglePhotonPrescaledL1MatchFilter.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1IsoSinglePhotonPrescaledL1MatchFilter.L1SeedFilterTag = 'l1seedSinglePrescaled'
hltL1IsoSinglePhotonPrescaledEtFilter.inputTag = 'hltL1IsoSinglePhotonPrescaledL1MatchFilter'
hltL1IsoSinglePhotonPrescaledEtFilter.etcut = 12.0
hltL1IsoSinglePhotonPrescaledEcalIsolFilter.candTag = 'hltL1IsoSinglePhotonPrescaledEtFilter'
hltL1IsoSinglePhotonPrescaledEcalIsolFilter.isoTag = 'l1IsolatedPhotonEcalIsol'
hltL1IsoSinglePhotonPrescaledHcalIsolFilter.candTag = 'hltL1IsoSinglePhotonPrescaledEcalIsolFilter'
hltL1IsoSinglePhotonPrescaledHcalIsolFilter.isoTag = 'l1IsolatedPhotonHcalIsol'
hltL1IsoSinglePhotonPrescaledHcalIsolFilter.hcalisolbarrelcut = 6.
hltL1IsoSinglePhotonPrescaledHcalIsolFilter.hcalisolendcapcut = 4.
hltL1IsoSinglePhotonPrescaledTrackIsolFilter.candTag = 'hltL1IsoSinglePhotonPrescaledHcalIsolFilter'
hltL1IsoSinglePhotonPrescaledTrackIsolFilter.isoTag = 'l1IsoPhotonTrackIsol'

