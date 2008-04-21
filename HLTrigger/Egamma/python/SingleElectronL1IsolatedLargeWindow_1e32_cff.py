# The following comments couldn't be translated into the new config version:

# pixel rechit for SC-pixel matching
# strip rechit for the electron tracking
#
import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1IsoLargeWindowSingleL1MatchFilter = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltEgammaEtFilter_cfi import *
#--------------------------------------------------------------------#
# ET CUT
#--------------------------------------------------------------------#
hltL1IsoLargeWindowSingleElectronEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#      int32  ncandcut = 1
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1IsoLargeWindowSingleElectronHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltElectronPixelMatchFilter_cfi import *
#      double hcalisolbarrelcut = 3.
#      double hcalisolendcapcut = 3.
#      int32 ncandcut = 1
#--------------------------------------------------------------------#
# PIXEL MATCH
#--------------------------------------------------------------------#
hltL1IsoLargeWindowSingleElectronPixelMatchFilter = copy.deepcopy(hltElectronPixelMatchFilter)
import copy
from HLTrigger.Egamma.hltElectronOneOEMinusOneOPFilter_cfi import *
#      double   npixelmatchcut         = 1
#      int32    ncandcut               = 1
# #--------------------------------------------------------------------#
# # E OVER P
# #--------------------------------------------------------------------#
#  module hltL1IsoLargeWindowSingleElectronEoverpFilter = hltElectronEoverpFilter from "HLTrigger/Egamma/data/hltElectronEoverpFilter.cfi"
# 
#  replace hltL1IsoLargeWindowSingleElectronEoverpFilter.candTag = hltL1IsoLargeWindowSingleElectronPixelMatchFilter
#  replace hltL1IsoLargeWindowSingleElectronEoverpFilter.electronIsolatedProducer = pixelMatchElectronsL1IsoLargeWindowForHLT 
# replace hltL1IsoLargeWindowSingleElectronEoverpFilter.doIsolated = true
# 
# #      double eoverpbarrelcut = 1.5
# #      double eoverpendcapcut = 2.45
# #      int32 ncandcut = 1
#--------------------------------------------------------------------#
# 1/E - 1/P
#--------------------------------------------------------------------#
hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter = copy.deepcopy(hltElectronOneOEMinusOneOPFilter)
import copy
from HLTrigger.Egamma.hltElectronTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1IsoLargeWindowSingleElectronTrackIsolFilter = copy.deepcopy(hltElectronTrackIsolFilter)
#      double pttrackisolcut = 0.06
#      int32 ncandcut         = 1
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
singleElectronL1IsoLargeWindowPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

singleElectronL1IsolatedLargeWindow = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedSingle")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+hltL1IsoLargeWindowSingleL1MatchFilter+hltL1IsoLargeWindowSingleElectronEtFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedElectronHcalIsol")+hltL1IsoLargeWindowSingleElectronHcalIsolFilter+cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("pixelMatchElectronL1IsoLargeWindowSequenceForHLT")+hltL1IsoLargeWindowSingleElectronPixelMatchFilter+cms.SequencePlaceholder("pixelMatchElectronL1IsoLargeWindowTrackingSequenceForHLT")+hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter+cms.SequencePlaceholder("l1IsoLargeWindowElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoLargeWindowElectronTrackIsol")+hltL1IsoLargeWindowSingleElectronTrackIsolFilter+singleElectronL1IsoLargeWindowPresc)
hltL1IsoLargeWindowSingleL1MatchFilter.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1IsoLargeWindowSingleL1MatchFilter.L1SeedFilterTag = 'l1seedSingle'
hltL1IsoLargeWindowSingleElectronEtFilter.inputTag = 'hltL1IsoLargeWindowSingleL1MatchFilter'
hltL1IsoLargeWindowSingleElectronEtFilter.etcut = 15.0
hltL1IsoLargeWindowSingleElectronHcalIsolFilter.candTag = 'hltL1IsoLargeWindowSingleElectronEtFilter'
hltL1IsoLargeWindowSingleElectronHcalIsolFilter.isoTag = 'l1IsolatedElectronHcalIsol'
hltL1IsoLargeWindowSingleElectronPixelMatchFilter.candTag = 'hltL1IsoLargeWindowSingleElectronHcalIsolFilter'
hltL1IsoLargeWindowSingleElectronPixelMatchFilter.L1IsoPixelSeedsTag = 'l1IsoLargeWindowElectronPixelSeeds'
hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.candTag = 'hltL1IsoLargeWindowSingleElectronPixelMatchFilter'
hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.electronIsolatedProducer = 'pixelMatchElectronsL1IsoLargeWindowForHLT'
hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.electronNonIsolatedProducer = 'pixelMatchElectronsL1NonIsoLargeWindowForHLT'
hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.barrelcut = 999.03
hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.endcapcut = 999.03
hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.ncandcut = 1
hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.doIsolated = True
hltL1IsoLargeWindowSingleElectronTrackIsolFilter.candTag = 'hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter'
hltL1IsoLargeWindowSingleElectronTrackIsolFilter.isoTag = 'l1IsoLargeWindowElectronTrackIsol'

