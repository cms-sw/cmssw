# The following comments couldn't be translated into the new config version:

#              doL1T &

#                             doLocalEcal & 

# trackerlocalreco &              

#ckftracks & 

# EgammaHLTRegionalPixelSeedGeneratorProducer

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
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
hltL1IsoLargeWindowDoubleElectronEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1IsoLargeWindowDoubleElectronHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltElectronPixelMatchFilter_cfi import *
#--------------------------------------------------------------------#
# PIXEL MATCH
#--------------------------------------------------------------------#
hltL1IsoLargeWindowDoubleElectronPixelMatchFilter = copy.deepcopy(hltElectronPixelMatchFilter)
import copy
from HLTrigger.Egamma.hltElectronEoverpFilter_cfi import *
#--------------------------------------------------------------------#
# E OVER P !! HERE is not applyed and it is used to change 
#       the HLTFilterObjectWithRefs from RecoEcalCandidates to Electron
#--------------------------------------------------------------------#
hltL1IsoLargeWindowDoubleElectronEoverpFilter = copy.deepcopy(hltElectronEoverpFilter)
import copy
from HLTrigger.Egamma.hltElectronTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1IsoLargeWindowDoubleElectronTrackIsolFilter = copy.deepcopy(hltElectronTrackIsolFilter)
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
doubleElectronL1IsoLargeWindowPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

doubleElectronL1IsolatedLargeWindow = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedDouble")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional+hltL1IsoLargeWindowDoubleElectronEtFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedElectronHcalIsol")+hltL1IsoLargeWindowDoubleElectronHcalIsolFilter+cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("pixelMatchElectronL1IsoLargeWindowSequenceForHLT")+hltL1IsoLargeWindowDoubleElectronPixelMatchFilter+cms.SequencePlaceholder("pixelMatchElectronL1IsoLargeWindowTrackingSequenceForHLT")+hltL1IsoLargeWindowDoubleElectronEoverpFilter+cms.SequencePlaceholder("l1IsoLargeWindowElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoLargeWindowElectronTrackIsol")+hltL1IsoLargeWindowDoubleElectronTrackIsolFilter+doubleElectronL1IsoLargeWindowPresc)
hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional.L1SeedFilterTag = 'l1seedDouble'
#     InputTag l1Tag = l1extraParticles:Isolated
hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional.ncandcut = 2
hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional.doIsolated = True
#replace hltL1IsoLargeWindowDoubleElectronEtFilter.inputTag = hltL1IsoLargeWindowDoubleElectronL1MatchFilter
hltL1IsoLargeWindowDoubleElectronEtFilter.inputTag = 'hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional'
hltL1IsoLargeWindowDoubleElectronEtFilter.etcut = 10.0
hltL1IsoLargeWindowDoubleElectronEtFilter.ncandcut = 2
hltL1IsoLargeWindowDoubleElectronHcalIsolFilter.candTag = 'hltL1IsoLargeWindowDoubleElectronEtFilter'
hltL1IsoLargeWindowDoubleElectronHcalIsolFilter.isoTag = 'l1IsolatedElectronHcalIsol'
hltL1IsoLargeWindowDoubleElectronHcalIsolFilter.hcalisolbarrelcut = 9.
hltL1IsoLargeWindowDoubleElectronHcalIsolFilter.hcalisolendcapcut = 9.
hltL1IsoLargeWindowDoubleElectronHcalIsolFilter.ncandcut = 2
hltL1IsoLargeWindowDoubleElectronPixelMatchFilter.candTag = 'hltL1IsoLargeWindowDoubleElectronHcalIsolFilter'
hltL1IsoLargeWindowDoubleElectronPixelMatchFilter.L1IsoPixelSeedsTag = 'l1IsoLargeWindowElectronPixelSeeds'
#      double   npixelmatchcut         = 1
hltL1IsoLargeWindowDoubleElectronPixelMatchFilter.ncandcut = 2
hltL1IsoLargeWindowDoubleElectronEoverpFilter.candTag = 'hltL1IsoLargeWindowDoubleElectronPixelMatchFilter'
hltL1IsoLargeWindowDoubleElectronEoverpFilter.electronIsolatedProducer = 'pixelMatchElectronsL1IsoLargeWindowForHLT'
hltL1IsoLargeWindowDoubleElectronEoverpFilter.eoverpbarrelcut = 15000
hltL1IsoLargeWindowDoubleElectronEoverpFilter.eoverpendcapcut = 24500
hltL1IsoLargeWindowDoubleElectronEoverpFilter.ncandcut = 2
hltL1IsoLargeWindowDoubleElectronTrackIsolFilter.candTag = 'hltL1IsoLargeWindowDoubleElectronEoverpFilter'
hltL1IsoLargeWindowDoubleElectronTrackIsolFilter.isoTag = 'l1IsoLargeWindowElectronTrackIsol'
hltL1IsoLargeWindowDoubleElectronTrackIsolFilter.pttrackisolcut = 0.4
hltL1IsoLargeWindowDoubleElectronTrackIsolFilter.ncandcut = 2

