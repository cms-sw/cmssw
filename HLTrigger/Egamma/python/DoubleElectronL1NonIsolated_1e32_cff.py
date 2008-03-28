# The following comments couldn't be translated into the new config version:

# doL1T &
# was like thisl1seedDouble &

# hltL1IsoDoubleElectronL1MatchFilter

#attention Iso
# trackerlocalreco &         

# ckftracks & 

# EgammaHLTRegionalPixelSeedGeneratorProducer

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1NonIsoDoubleElectronL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
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
hltL1NonIsoDoubleElectronEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1NonIsoDoubleElectronHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltElectronPixelMatchFilter_cfi import *
#--------------------------------------------------------------------#
# PIXEL MATCH
#--------------------------------------------------------------------#
hltL1NonIsoDoubleElectronPixelMatchFilter = copy.deepcopy(hltElectronPixelMatchFilter)
import copy
from HLTrigger.Egamma.hltElectronEoverpFilter_cfi import *
#--------------------------------------------------------------------#
# E OVER P !! HERE is not applyed and it is used to change 
#       the HLTFilterObjectWithRefs from RecoEcalCandidates to Electron
#--------------------------------------------------------------------#
hltL1NonIsoDoubleElectronEoverpFilter = copy.deepcopy(hltElectronEoverpFilter)
import copy
from HLTrigger.Egamma.hltElectronTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1NonIsoDoubleElectronTrackIsolFilter = copy.deepcopy(hltElectronTrackIsolFilter)
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
doubleElectronL1NonIsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

doubleElectronL1NonIsolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedRelaxedDouble")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1NonIsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+cms.SequencePlaceholder("l1NonIsoRecoEcalCandidate")+hltL1NonIsoDoubleElectronL1MatchFilterRegional+hltL1NonIsoDoubleElectronEtFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedElectronHcalIsol")+cms.SequencePlaceholder("l1NonIsolatedElectronHcalIsol")+hltL1NonIsoDoubleElectronHcalIsolFilter+cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("pixelMatchElectronL1IsoSequenceForHLT")+cms.SequencePlaceholder("pixelMatchElectronL1NonIsoSequenceForHLT")+hltL1NonIsoDoubleElectronPixelMatchFilter+cms.SequencePlaceholder("pixelMatchElectronL1IsoTrackingSequenceForHLT")+cms.SequencePlaceholder("pixelMatchElectronL1NonIsoTrackingSequenceForHLT")+hltL1NonIsoDoubleElectronEoverpFilter+cms.SequencePlaceholder("l1IsoElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1NonIsoElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoElectronTrackIsol")+cms.SequencePlaceholder("l1NonIsoElectronTrackIsol")+hltL1NonIsoDoubleElectronTrackIsolFilter+doubleElectronL1NonIsoPresc)
hltL1NonIsoDoubleElectronL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1NonIsoDoubleElectronL1MatchFilterRegional.candNonIsolatedTag = 'l1NonIsoRecoEcalCandidate'
hltL1NonIsoDoubleElectronL1MatchFilterRegional.L1SeedFilterTag = 'l1seedRelaxedDouble'
#     InputTag l1Tag = l1extraParticles:Isolated
hltL1NonIsoDoubleElectronL1MatchFilterRegional.ncandcut = 2
hltL1NonIsoDoubleElectronL1MatchFilterRegional.doIsolated = False
hltL1NonIsoDoubleElectronEtFilter.inputTag = 'hltL1NonIsoDoubleElectronL1MatchFilterRegional'
#replace hltL1IsoDoubleElectronEtFilter.inputTag = hltL1IsoDoubleElectronL1MatchFilter
hltL1NonIsoDoubleElectronEtFilter.etcut = 12.0
hltL1NonIsoDoubleElectronEtFilter.ncandcut = 2
hltL1NonIsoDoubleElectronHcalIsolFilter.candTag = 'hltL1NonIsoDoubleElectronEtFilter'
hltL1NonIsoDoubleElectronHcalIsolFilter.isoTag = 'l1IsolatedElectronHcalIsol'
hltL1NonIsoDoubleElectronHcalIsolFilter.nonIsoTag = 'l1NonIsolatedElectronHcalIsol'
hltL1NonIsoDoubleElectronHcalIsolFilter.hcalisolbarrelcut = 9.
hltL1NonIsoDoubleElectronHcalIsolFilter.hcalisolendcapcut = 9.
hltL1NonIsoDoubleElectronHcalIsolFilter.ncandcut = 2
hltL1NonIsoDoubleElectronHcalIsolFilter.doIsolated = False
hltL1NonIsoDoubleElectronPixelMatchFilter.candTag = 'hltL1NonIsoDoubleElectronHcalIsolFilter'
hltL1NonIsoDoubleElectronPixelMatchFilter.L1IsoPixelSeedsTag = 'l1IsoElectronPixelSeeds'
# replace hltL1NonIsoDoubleElectronPixelMatchFilter.L1IsoPixelmapendcapTag = l1IsoElectronPixelSeeds:correctedEndcapSuperClustersWithPreshowerL1Isolated
hltL1NonIsoDoubleElectronPixelMatchFilter.L1NonIsoPixelSeedsTag = 'l1NonIsoElectronPixelSeeds'
#  replace hltL1NonIsoDoubleElectronPixelMatchFilter.L1NonIsoPixelmapendcapTag = l1NonIsoElectronPixelSeeds:correctedEndcapSuperClustersWithPreshowerL1NonIsolated
hltL1NonIsoDoubleElectronPixelMatchFilter.doIsolated = False
#      double   npixelmatchcut         = 1
hltL1NonIsoDoubleElectronPixelMatchFilter.ncandcut = 2
hltL1NonIsoDoubleElectronEoverpFilter.candTag = 'hltL1NonIsoDoubleElectronPixelMatchFilter'
hltL1NonIsoDoubleElectronEoverpFilter.electronIsolatedProducer = 'pixelMatchElectronsL1IsoForHLT'
hltL1NonIsoDoubleElectronEoverpFilter.electronNonIsolatedProducer = 'pixelMatchElectronsL1NonIsoForHLT'
hltL1NonIsoDoubleElectronEoverpFilter.eoverpbarrelcut = 15000
hltL1NonIsoDoubleElectronEoverpFilter.eoverpendcapcut = 24500
hltL1NonIsoDoubleElectronEoverpFilter.ncandcut = 2
hltL1NonIsoDoubleElectronEoverpFilter.doIsolated = False
hltL1NonIsoDoubleElectronTrackIsolFilter.candTag = 'hltL1NonIsoDoubleElectronEoverpFilter'
hltL1NonIsoDoubleElectronTrackIsolFilter.isoTag = 'l1IsoElectronTrackIsol'
hltL1NonIsoDoubleElectronTrackIsolFilter.nonIsoTag = 'l1NonIsoElectronTrackIsol'
hltL1NonIsoDoubleElectronTrackIsolFilter.pttrackisolcut = 0.4
hltL1NonIsoDoubleElectronTrackIsolFilter.ncandcut = 2
hltL1NonIsoDoubleElectronTrackIsolFilter.doIsolated = False

