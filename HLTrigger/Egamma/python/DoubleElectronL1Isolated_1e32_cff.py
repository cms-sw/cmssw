# The following comments couldn't be translated into the new config version:

#              doL1T &

#                             doLocalEcal & 

#hltL1IsoDoubleElectronL1MatchFilter &

# trackerlocalreco &              

#ckftracks & 

# EgammaHLTRegionalPixelSeedGeneratorProducer

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1IsoDoubleElectronL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltEgammaEtFilter_cfi import *
#
#      double region_eta_size      = 0.522
#      double region_eta_size_ecap = 0.87
#      double region_phi_size      = 1.044
#      double barrel_end           = 1.4791
#      double endcap_end           = 2.5
#--------------------------------------------------------------------#
# ET CUT
#--------------------------------------------------------------------#
hltL1IsoDoubleElectronEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1IsoDoubleElectronHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltElectronPixelMatchFilter_cfi import *
#--------------------------------------------------------------------#
# PIXEL MATCH
#--------------------------------------------------------------------#
hltL1IsoDoubleElectronPixelMatchFilter = copy.deepcopy(hltElectronPixelMatchFilter)
import copy
from HLTrigger.Egamma.hltElectronEoverpFilter_cfi import *
#--------------------------------------------------------------------#
# E OVER P !! HERE is not applyed and it is used to change 
#       the HLTFilterObjectWithRefs from RecoEcalCandidates to Electron
#--------------------------------------------------------------------#
hltL1IsoDoubleElectronEoverpFilter = copy.deepcopy(hltElectronEoverpFilter)
import copy
from HLTrigger.Egamma.hltElectronTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1IsoDoubleElectronTrackIsolFilter = copy.deepcopy(hltElectronTrackIsolFilter)
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
doubleElectronL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

doubleElectronL1Isolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedDouble")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+hltL1IsoDoubleElectronL1MatchFilterRegional+hltL1IsoDoubleElectronEtFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedElectronHcalIsol")+hltL1IsoDoubleElectronHcalIsolFilter+cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("pixelMatchElectronL1IsoSequenceForHLT")+hltL1IsoDoubleElectronPixelMatchFilter+cms.SequencePlaceholder("pixelMatchElectronL1IsoTrackingSequenceForHLT")+hltL1IsoDoubleElectronEoverpFilter+cms.SequencePlaceholder("l1IsoElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoElectronTrackIsol")+hltL1IsoDoubleElectronTrackIsolFilter+doubleElectronL1IsoPresc)
hltL1IsoDoubleElectronL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1IsoDoubleElectronL1MatchFilterRegional.L1SeedFilterTag = 'l1seedDouble'
#     InputTag l1Tag = l1extraParticles:Isolated
#replace  hltL1IsoDoubleElectronL1MatchFilter.ncandcut = 2
hltL1IsoDoubleElectronL1MatchFilterRegional.ncandcut = 2
hltL1IsoDoubleElectronL1MatchFilterRegional.doIsolated = True
#replace hltL1IsoDoubleElectronEtFilter.inputTag = hltL1IsoDoubleElectronL1MatchFilter
hltL1IsoDoubleElectronEtFilter.inputTag = 'hltL1IsoDoubleElectronL1MatchFilterRegional'
hltL1IsoDoubleElectronEtFilter.etcut = 10.0
hltL1IsoDoubleElectronEtFilter.ncandcut = 2
hltL1IsoDoubleElectronHcalIsolFilter.candTag = 'hltL1IsoDoubleElectronEtFilter'
hltL1IsoDoubleElectronHcalIsolFilter.isoTag = 'l1IsolatedElectronHcalIsol'
hltL1IsoDoubleElectronHcalIsolFilter.hcalisolbarrelcut = 9.
hltL1IsoDoubleElectronHcalIsolFilter.hcalisolendcapcut = 9.
hltL1IsoDoubleElectronHcalIsolFilter.ncandcut = 2
hltL1IsoDoubleElectronPixelMatchFilter.candTag = 'hltL1IsoDoubleElectronHcalIsolFilter'
hltL1IsoDoubleElectronPixelMatchFilter.L1IsoPixelSeedsTag = 'l1IsoElectronPixelSeeds'
#      double   npixelmatchcut         = 1
hltL1IsoDoubleElectronPixelMatchFilter.ncandcut = 2
hltL1IsoDoubleElectronEoverpFilter.candTag = 'hltL1IsoDoubleElectronPixelMatchFilter'
hltL1IsoDoubleElectronEoverpFilter.electronIsolatedProducer = 'pixelMatchElectronsL1IsoForHLT'
hltL1IsoDoubleElectronEoverpFilter.eoverpbarrelcut = 15000
hltL1IsoDoubleElectronEoverpFilter.eoverpendcapcut = 24500
hltL1IsoDoubleElectronEoverpFilter.ncandcut = 2
hltL1IsoDoubleElectronTrackIsolFilter.candTag = 'hltL1IsoDoubleElectronEoverpFilter'
hltL1IsoDoubleElectronTrackIsolFilter.isoTag = 'l1IsoElectronTrackIsol'
hltL1IsoDoubleElectronTrackIsolFilter.pttrackisolcut = 0.4
hltL1IsoDoubleElectronTrackIsolFilter.ncandcut = 2

