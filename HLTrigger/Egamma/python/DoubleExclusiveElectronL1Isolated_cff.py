# The following comments couldn't be translated into the new config version:

#                                      doL1T &

#
# EgammaHLTRegionalPixelSeedGeneratorProducer

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1IsoDoubleExclElectronL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltDoubleExclusiveElectronFilter_cfi import *
#--------------------------------------------------------------------#
# ET AND PHI CUT
#--------------------------------------------------------------------#
hltL1IsoDoubleExclElectronEtPhiFilter = copy.deepcopy(hltDoubleExclusiveElectronFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1IsoDoubleExclElectronHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltElectronPixelMatchFilter_cfi import *
#--------------------------------------------------------------------#
# PIXEL MATCH
#--------------------------------------------------------------------#
hltL1IsoDoubleExclElectronPixelMatchFilter = copy.deepcopy(hltElectronPixelMatchFilter)
import copy
from HLTrigger.Egamma.hltElectronEoverpFilter_cfi import *
#--------------------------------------------------------------------#
# E OVER P !! HERE is not applyed and it is used to change 
#       the HLTFilterObjectWithRefs from RecoEcalCandidates to Electron
#--------------------------------------------------------------------#
hltL1IsoDoubleExclElectronEoverpFilter = copy.deepcopy(hltElectronEoverpFilter)
import copy
from HLTrigger.Egamma.hltElectronTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1IsoDoubleExclElectronTrackIsolFilter = copy.deepcopy(hltElectronTrackIsolFilter)
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
doubleExclElectronL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

doubleExclElectronL1Isolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedExclusiveDouble")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+hltL1IsoDoubleExclElectronL1MatchFilterRegional+hltL1IsoDoubleExclElectronEtPhiFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedElectronHcalIsol")+hltL1IsoDoubleExclElectronHcalIsolFilter+cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("pixelMatchElectronL1IsoSequenceForHLT")+hltL1IsoDoubleExclElectronPixelMatchFilter+cms.SequencePlaceholder("pixelMatchElectronL1IsoTrackingSequenceForHLT")+hltL1IsoDoubleExclElectronEoverpFilter+cms.SequencePlaceholder("l1IsoElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoElectronTrackIsol")+hltL1IsoDoubleExclElectronTrackIsolFilter+doubleExclElectronL1IsoPresc)
hltL1IsoDoubleExclElectronL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1IsoDoubleExclElectronL1MatchFilterRegional.L1SeedFilterTag = 'l1seedExclusiveDouble'
hltL1IsoDoubleExclElectronL1MatchFilterRegional.ncandcut = 2
hltL1IsoDoubleExclElectronL1MatchFilterRegional.doIsolated = True
hltL1IsoDoubleExclElectronEtPhiFilter.candTag = 'hltL1IsoDoubleExclElectronL1MatchFilterRegional'
hltL1IsoDoubleExclElectronHcalIsolFilter.candTag = 'hltL1IsoDoubleExclElectronEtPhiFilter'
hltL1IsoDoubleExclElectronHcalIsolFilter.isoTag = 'l1IsolatedElectronHcalIsol'
hltL1IsoDoubleExclElectronHcalIsolFilter.hcalisolbarrelcut = 9.
hltL1IsoDoubleExclElectronHcalIsolFilter.hcalisolendcapcut = 9.
hltL1IsoDoubleExclElectronHcalIsolFilter.ncandcut = 2
hltL1IsoDoubleExclElectronPixelMatchFilter.candTag = 'hltL1IsoDoubleExclElectronHcalIsolFilter'
hltL1IsoDoubleExclElectronPixelMatchFilter.L1IsoPixelSeedsTag = 'l1IsoElectronPixelSeeds'
hltL1IsoDoubleExclElectronPixelMatchFilter.ncandcut = 2
hltL1IsoDoubleExclElectronEoverpFilter.candTag = 'hltL1IsoDoubleExclElectronPixelMatchFilter'
hltL1IsoDoubleExclElectronEoverpFilter.electronIsolatedProducer = 'pixelMatchElectronsL1IsoForHLT'
hltL1IsoDoubleExclElectronEoverpFilter.eoverpbarrelcut = 15000
hltL1IsoDoubleExclElectronEoverpFilter.eoverpendcapcut = 24500
hltL1IsoDoubleExclElectronEoverpFilter.ncandcut = 2
hltL1IsoDoubleExclElectronTrackIsolFilter.candTag = 'hltL1IsoDoubleExclElectronEoverpFilter'
hltL1IsoDoubleExclElectronTrackIsolFilter.isoTag = 'l1IsoElectronTrackIsol'
hltL1IsoDoubleExclElectronTrackIsolFilter.pttrackisolcut = 0.4
hltL1IsoDoubleExclElectronTrackIsolFilter.ncandcut = 2

