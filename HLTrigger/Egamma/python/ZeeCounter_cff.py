# The following comments couldn't be translated into the new config version:

#              doL1T &

#                             doLocalEcal & 

#hltL1IsoDoubleElectronZeeL1MatchFilter &

# trackerlocalreco &              

#ckftracks & 

# EgammaHLTRegionalPixelSeedGeneratorProducer

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1IsoDoubleElectronZeeL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltEgammaEtFilter_cfi import *
#--------------------------------------------------------------------#
# ET CUT
#--------------------------------------------------------------------#
hltL1IsoDoubleElectronZeeEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1IsoDoubleElectronZeeHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltElectronPixelMatchFilter_cfi import *
#--------------------------------------------------------------------#
# PIXEL MATCH
#--------------------------------------------------------------------#
hltL1IsoDoubleElectronZeePixelMatchFilter = copy.deepcopy(hltElectronPixelMatchFilter)
import copy
from HLTrigger.Egamma.hltElectronEoverpFilter_cfi import *
#--------------------------------------------------------------------#
# E OVER P !! HERE is not applyed and it is used to change 
#       the HLTFilterObjectWithRefs from RecoEcalCandidates to Electron
#--------------------------------------------------------------------#
hltL1IsoDoubleElectronZeeEoverpFilter = copy.deepcopy(hltElectronEoverpFilter)
import copy
from HLTrigger.Egamma.hltElectronTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1IsoDoubleElectronZeeTrackIsolFilter = copy.deepcopy(hltElectronTrackIsolFilter)
import copy
from HLTrigger.Egamma.hltZeePMMassFilter_cfi import *
#--------------------------------------------------------------------#
# INV MASS
#--------------------------------------------------------------------#
hltL1IsoDoubleElectronZeePMMassFilter = copy.deepcopy(hltZeePMMassFilter)
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
zeeCounterPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

zeeCounter = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedDouble")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+hltL1IsoDoubleElectronZeeL1MatchFilterRegional+hltL1IsoDoubleElectronZeeEtFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedElectronHcalIsol")+hltL1IsoDoubleElectronZeeHcalIsolFilter+cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("pixelMatchElectronL1IsoSequenceForHLT")+hltL1IsoDoubleElectronZeePixelMatchFilter+cms.SequencePlaceholder("pixelMatchElectronL1IsoTrackingSequenceForHLT")+hltL1IsoDoubleElectronZeeEoverpFilter+cms.SequencePlaceholder("l1IsoElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoElectronTrackIsol")+hltL1IsoDoubleElectronZeeTrackIsolFilter+hltL1IsoDoubleElectronZeePMMassFilter+zeeCounterPresc)
hltL1IsoDoubleElectronZeeL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1IsoDoubleElectronZeeL1MatchFilterRegional.L1SeedFilterTag = 'l1seedDouble'
hltL1IsoDoubleElectronZeeL1MatchFilterRegional.ncandcut = 2
hltL1IsoDoubleElectronZeeL1MatchFilterRegional.doIsolated = True
#replace hltL1IsoDoubleElectronZeeEtFilter.inputTag = hltL1IsoDoubleElectronZeeL1MatchFilter
hltL1IsoDoubleElectronZeeEtFilter.inputTag = 'hltL1IsoDoubleElectronZeeL1MatchFilterRegional'
hltL1IsoDoubleElectronZeeEtFilter.etcut = 10.0
hltL1IsoDoubleElectronZeeEtFilter.ncandcut = 2
hltL1IsoDoubleElectronZeeHcalIsolFilter.candTag = 'hltL1IsoDoubleElectronZeeEtFilter'
hltL1IsoDoubleElectronZeeHcalIsolFilter.isoTag = 'l1IsolatedElectronHcalIsol'
hltL1IsoDoubleElectronZeeHcalIsolFilter.hcalisolbarrelcut = 9.
hltL1IsoDoubleElectronZeeHcalIsolFilter.hcalisolendcapcut = 9.
hltL1IsoDoubleElectronZeeHcalIsolFilter.ncandcut = 2
hltL1IsoDoubleElectronZeePixelMatchFilter.candTag = 'hltL1IsoDoubleElectronZeeHcalIsolFilter'
hltL1IsoDoubleElectronZeePixelMatchFilter.L1IsoPixelSeedsTag = 'l1IsoElectronPixelSeeds'
#  replace hltL1IsoDoubleElectronZeePixelMatchFilter.L1IsoPixelmapendcapTag = l1IsoElectronPixelSeeds:correctedEndcapSuperClustersWithPreshowerL1Isolated
#      double   npixelmatchcut         = 1
hltL1IsoDoubleElectronZeePixelMatchFilter.ncandcut = 2
hltL1IsoDoubleElectronZeeEoverpFilter.candTag = 'hltL1IsoDoubleElectronZeePixelMatchFilter'
hltL1IsoDoubleElectronZeeEoverpFilter.electronIsolatedProducer = 'pixelMatchElectronsL1IsoForHLT'
hltL1IsoDoubleElectronZeeEoverpFilter.eoverpbarrelcut = 15000
hltL1IsoDoubleElectronZeeEoverpFilter.eoverpendcapcut = 24500
hltL1IsoDoubleElectronZeeEoverpFilter.ncandcut = 2
hltL1IsoDoubleElectronZeeTrackIsolFilter.candTag = 'hltL1IsoDoubleElectronZeeEoverpFilter'
hltL1IsoDoubleElectronZeeTrackIsolFilter.isoTag = 'l1IsoElectronTrackIsol'
hltL1IsoDoubleElectronZeeTrackIsolFilter.pttrackisolcut = 0.4
hltL1IsoDoubleElectronZeeTrackIsolFilter.ncandcut = 2
hltL1IsoDoubleElectronZeePMMassFilter.candTag = 'hltL1IsoDoubleElectronZeeTrackIsolFilter'
hltL1IsoDoubleElectronZeePMMassFilter.lowerMassCut = 54.22
hltL1IsoDoubleElectronZeePMMassFilter.upperMassCut = 99999.9
hltL1IsoDoubleElectronZeePMMassFilter.nZcandcut = 1

