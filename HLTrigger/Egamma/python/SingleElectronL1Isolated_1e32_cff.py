# The following comments couldn't be translated into the new config version:

#hltL1IsoSingleElectronEoverpFilter &	

# EgammaHLTRegionalPixelSeedGeneratorProducer

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1IsoSingleL1MatchFilter = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltEgammaEtFilter_cfi import *
#--------------------------------------------------------------------#
# ET CUT
#--------------------------------------------------------------------#
hltL1IsoSingleElectronEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#      int32  ncandcut = 1
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1IsoSingleElectronHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltElectronPixelMatchFilter_cfi import *
#      double hcalisolbarrelcut = 3.
#      double hcalisolendcapcut = 3.
#      int32 ncandcut = 1
#--------------------------------------------------------------------#
# PIXEL MATCH
#--------------------------------------------------------------------#
hltL1IsoSingleElectronPixelMatchFilter = copy.deepcopy(hltElectronPixelMatchFilter)
import copy
from HLTrigger.Egamma.hltElectronOneOEMinusOneOPFilter_cfi import *
#      double   npixelmatchcut         = 1
#      int32    ncandcut               = 1
# #--------------------------------------------------------------------#
# # E OVER P
# #--------------------------------------------------------------------#
#  module hltL1IsoSingleElectronEoverpFilter = hltElectronEoverpFilter from "HLTrigger/Egamma/data/hltElectronEoverpFilter.cfi"
# 
#  replace hltL1IsoSingleElectronEoverpFilter.candTag = hltL1IsoSingleElectronPixelMatchFilter
#  replace hltL1IsoSingleElectronEoverpFilter.electronIsolatedProducer = pixelMatchElectronsL1IsoForHLT 
# replace hltL1IsoSingleElectronEoverpFilter.doIsolated = true
# 
# #      double eoverpbarrelcut = 1.5
# #      double eoverpendcapcut = 2.45
# #      int32 ncandcut = 1
#--------------------------------------------------------------------#
# 1/E - 1/P
#--------------------------------------------------------------------#
hltL1IsoSingleElectronHOneOEMinusOneOPFilter = copy.deepcopy(hltElectronOneOEMinusOneOPFilter)
import copy
from HLTrigger.Egamma.hltElectronTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1IsoSingleElectronTrackIsolFilter = copy.deepcopy(hltElectronTrackIsolFilter)
#      double pttrackisolcut = 0.06
#      int32 ncandcut         = 1
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
singleElectronL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

singleElectronL1Isolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedSingle")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+hltL1IsoSingleL1MatchFilter+hltL1IsoSingleElectronEtFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedElectronHcalIsol")+hltL1IsoSingleElectronHcalIsolFilter+cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("pixelMatchElectronL1IsoSequenceForHLT")+hltL1IsoSingleElectronPixelMatchFilter+cms.SequencePlaceholder("pixelMatchElectronL1IsoTrackingSequenceForHLT")+hltL1IsoSingleElectronHOneOEMinusOneOPFilter+cms.SequencePlaceholder("l1IsoElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoElectronTrackIsol")+hltL1IsoSingleElectronTrackIsolFilter+singleElectronL1IsoPresc)
hltL1IsoSingleL1MatchFilter.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1IsoSingleL1MatchFilter.L1SeedFilterTag = 'l1seedSingle'
hltL1IsoSingleElectronEtFilter.inputTag = 'hltL1IsoSingleL1MatchFilter'
hltL1IsoSingleElectronEtFilter.etcut = 15.0
hltL1IsoSingleElectronHcalIsolFilter.candTag = 'hltL1IsoSingleElectronEtFilter'
hltL1IsoSingleElectronHcalIsolFilter.isoTag = 'l1IsolatedElectronHcalIsol'
hltL1IsoSingleElectronPixelMatchFilter.candTag = 'hltL1IsoSingleElectronHcalIsolFilter'
hltL1IsoSingleElectronPixelMatchFilter.L1IsoPixelSeedsTag = 'l1IsoElectronPixelSeeds'
hltL1IsoSingleElectronHOneOEMinusOneOPFilter.candTag = 'hltL1IsoSingleElectronPixelMatchFilter'
hltL1IsoSingleElectronHOneOEMinusOneOPFilter.electronIsolatedProducer = 'pixelMatchElectronsL1IsoForHLT'
hltL1IsoSingleElectronHOneOEMinusOneOPFilter.electronNonIsolatedProducer = 'pixelMatchElectronsL1NonIsoForHLT'
hltL1IsoSingleElectronHOneOEMinusOneOPFilter.barrelcut = 999.03
hltL1IsoSingleElectronHOneOEMinusOneOPFilter.endcapcut = 999.03
hltL1IsoSingleElectronHOneOEMinusOneOPFilter.ncandcut = 1
hltL1IsoSingleElectronHOneOEMinusOneOPFilter.doIsolated = True
#replace  hltL1IsoSingleElectronTrackIsolFilter.candTag  = hltL1IsoSingleElectronEoverpFilter
hltL1IsoSingleElectronTrackIsolFilter.candTag = 'hltL1IsoSingleElectronHOneOEMinusOneOPFilter'
hltL1IsoSingleElectronTrackIsolFilter.isoTag = 'l1IsoElectronTrackIsol'

