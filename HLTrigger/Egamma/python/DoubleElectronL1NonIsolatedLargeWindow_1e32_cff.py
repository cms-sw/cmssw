# The following comments couldn't be translated into the new config version:

# doL1T &
# was like thisl1seedDouble &

#attention Iso
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
hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
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
hltL1NonIsoLargeWindowDoubleElectronEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltElectronPixelMatchFilter_cfi import *
#--------------------------------------------------------------------#
# PIXEL MATCH
#--------------------------------------------------------------------#
hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter = copy.deepcopy(hltElectronPixelMatchFilter)
import copy
from HLTrigger.Egamma.hltElectronEoverpFilter_cfi import *
#--------------------------------------------------------------------#
# E OVER P !! HERE is not applyed and it is used to change 
#       the HLTFilterObjectWithRefs from RecoEcalCandidates to Electron
#--------------------------------------------------------------------#
hltL1NonIsoLargeWindowDoubleElectronEoverpFilter = copy.deepcopy(hltElectronEoverpFilter)
import copy
from HLTrigger.Egamma.hltElectronTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter = copy.deepcopy(hltElectronTrackIsolFilter)
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
doubleElectronL1NonIsoLargeWindowPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

doubleElectronL1NonIsolatedLargeWindow = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedRelaxedDouble")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1NonIsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+cms.SequencePlaceholder("l1NonIsoRecoEcalCandidate")+hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional+hltL1NonIsoLargeWindowDoubleElectronEtFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedElectronHcalIsol")+cms.SequencePlaceholder("l1NonIsolatedElectronHcalIsol")+hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter+cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("pixelMatchElectronL1IsoLargeWindowSequenceForHLT")+cms.SequencePlaceholder("pixelMatchElectronL1NonIsoLargeWindowSequenceForHLT")+hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("pixelMatchElectronL1IsoLargeWindowTrackingSequenceForHLT")+cms.SequencePlaceholder("pixelMatchElectronL1NonIsoLargeWindowTrackingSequenceForHLT")+hltL1NonIsoLargeWindowDoubleElectronEoverpFilter+cms.SequencePlaceholder("l1IsoLargeWindowElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1NonIsoLargeWindowElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoLargeWindowElectronTrackIsol")+cms.SequencePlaceholder("l1NonIsoLargeWindowElectronTrackIsol")+hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter+doubleElectronL1NonIsoLargeWindowPresc)
hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional.candNonIsolatedTag = 'l1NonIsoRecoEcalCandidate'
hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional.L1SeedFilterTag = 'l1seedRelaxedDouble'
#     InputTag l1Tag = l1extraParticles:Isolated
hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional.ncandcut = 2
hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional.doIsolated = False
hltL1NonIsoLargeWindowDoubleElectronEtFilter.inputTag = 'hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional'
hltL1NonIsoLargeWindowDoubleElectronEtFilter.etcut = 12.0
hltL1NonIsoLargeWindowDoubleElectronEtFilter.ncandcut = 2
hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter.candTag = 'hltL1NonIsoLargeWindowDoubleElectronEtFilter'
hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter.isoTag = 'l1IsolatedElectronHcalIsol'
hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter.nonIsoTag = 'l1NonIsolatedElectronHcalIsol'
hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter.hcalisolbarrelcut = 9.
hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter.hcalisolendcapcut = 9.
hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter.ncandcut = 2
hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter.doIsolated = False
hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter.candTag = 'hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter'
hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter.L1IsoPixelSeedsTag = 'l1IsoLargeWindowElectronPixelSeeds'
hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter.L1NonIsoPixelSeedsTag = 'l1NonIsoLargeWindowElectronPixelSeeds'
hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter.doIsolated = False
#      double   npixelmatchcut         = 1
hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter.ncandcut = 2
hltL1NonIsoLargeWindowDoubleElectronEoverpFilter.candTag = 'hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter'
hltL1NonIsoLargeWindowDoubleElectronEoverpFilter.electronIsolatedProducer = 'pixelMatchElectronsL1IsoLargeWindowForHLT'
hltL1NonIsoLargeWindowDoubleElectronEoverpFilter.electronNonIsolatedProducer = 'pixelMatchElectronsL1NonIsoLargeWindowForHLT'
hltL1NonIsoLargeWindowDoubleElectronEoverpFilter.eoverpbarrelcut = 15000
hltL1NonIsoLargeWindowDoubleElectronEoverpFilter.eoverpendcapcut = 24500
hltL1NonIsoLargeWindowDoubleElectronEoverpFilter.ncandcut = 2
hltL1NonIsoLargeWindowDoubleElectronEoverpFilter.doIsolated = False
hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter.candTag = 'hltL1NonIsoLargeWindowDoubleElectronEoverpFilter'
hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter.isoTag = 'l1IsoLargeWindowElectronTrackIsol'
hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter.nonIsoTag = 'l1NonIsoLargeWindowElectronTrackIsol'
hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter.pttrackisolcut = 0.4
hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter.ncandcut = 2
hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter.doIsolated = False

