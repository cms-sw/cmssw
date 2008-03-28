# The following comments couldn't be translated into the new config version:

#doL1T &

#	                  doLocalEcal &

# trackerlocalreco &              

#ckftracks & 

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
#      double etcut = 26.0
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
from HLTrigger.Egamma.hltElectronEoverpFilter_cfi import *
#      double   npixelmatchcut         = 1
#      int32    ncandcut               = 1
#--------------------------------------------------------------------#
# E OVER P
#--------------------------------------------------------------------#
hltL1IsoSingleElectronEoverpFilter = copy.deepcopy(hltElectronEoverpFilter)
import copy
from HLTrigger.Egamma.hltElectronTrackIsolFilter_cfi import *
#      double eoverpbarrelcut = 1.5
#      double eoverpendcapcut = 2.45
#      int32 ncandcut = 1
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

singleElectronL1Isolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedSingle")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+hltL1IsoSingleL1MatchFilter+hltL1IsoSingleElectronEtFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedElectronHcalIsol")+hltL1IsoSingleElectronHcalIsolFilter+cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("pixelMatchElectronL1IsoSequenceForHLT")+hltL1IsoSingleElectronPixelMatchFilter+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("pixelMatchElectronL1IsoTrackingSequenceForHLT")+hltL1IsoSingleElectronEoverpFilter+cms.SequencePlaceholder("l1IsoElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoElectronTrackIsol")+hltL1IsoSingleElectronTrackIsolFilter+singleElectronL1IsoPresc)
hltL1IsoSingleL1MatchFilter.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1IsoSingleL1MatchFilter.L1SeedFilterTag = 'l1seedSingle'
hltL1IsoSingleElectronEtFilter.inputTag = 'hltL1IsoSingleL1MatchFilter'
hltL1IsoSingleElectronHcalIsolFilter.candTag = 'hltL1IsoSingleElectronEtFilter'
hltL1IsoSingleElectronHcalIsolFilter.isoTag = 'l1IsolatedElectronHcalIsol'
hltL1IsoSingleElectronPixelMatchFilter.candTag = 'hltL1IsoSingleElectronHcalIsolFilter'
hltL1IsoSingleElectronPixelMatchFilter.L1IsoPixelSeedsTag = 'l1IsoElectronPixelSeeds'
hltL1IsoSingleElectronEoverpFilter.candTag = 'hltL1IsoSingleElectronPixelMatchFilter'
hltL1IsoSingleElectronEoverpFilter.electronIsolatedProducer = 'pixelMatchElectronsL1IsoForHLT'
hltL1IsoSingleElectronEoverpFilter.doIsolated = True
hltL1IsoSingleElectronTrackIsolFilter.candTag = 'hltL1IsoSingleElectronEoverpFilter'
hltL1IsoSingleElectronTrackIsolFilter.isoTag = 'l1IsoElectronTrackIsol'

