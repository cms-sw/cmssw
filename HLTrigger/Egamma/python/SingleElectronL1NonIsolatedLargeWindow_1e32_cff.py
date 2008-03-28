# The following comments couldn't be translated into the new config version:

#hltL1NonIsoLargeWindowSingleElectronEoverpFilter &

# EgammaHLTRegionalPixelSeedGeneratorProducer

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
import copy
from HLTrigger.Egamma.hltEgammaEtFilter_cfi import *
#
#      double region_eta_size      = 0.522
#      double region_eta_size_ecap = 1.0
#      double region_phi_size      = 1.044
#      double barrel_end           = 1.4791
#      double endcap_end           = 2.65
#--------------------------------------------------------------------#
# ET CUT
#--------------------------------------------------------------------#
hltL1NonIsoLargeWindowSingleElectronEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltElectronPixelMatchFilter_cfi import *
#--------------------------------------------------------------------#
# PIXEL MATCH
#--------------------------------------------------------------------#
hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter = copy.deepcopy(hltElectronPixelMatchFilter)
import copy
from HLTrigger.Egamma.hltElectronOneOEMinusOneOPFilter_cfi import *
# #--------------------------------------------------------------------#
# # E OVER P       
# #--------------------------------------------------------------------#
#  module hltL1NonIsoLargeWindowSingleElectronEoverpFilter = hltElectronEoverpFilter from "HLTrigger/Egamma/data/hltElectronEoverpFilter.cfi"
# 
#  replace hltL1NonIsoLargeWindowSingleElectronEoverpFilter.candTag = hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter
#  replace hltL1NonIsoLargeWindowSingleElectronEoverpFilter.electronIsolatedProducer = pixelMatchElectronsL1IsoLargeWindowForHLT 
#  replace hltL1NonIsoLargeWindowSingleElectronEoverpFilter.electronNonIsolatedProducer = pixelMatchElectronsL1NonIsoLargeWindowForHLT 
#   replace hltL1NonIsoLargeWindowSingleElectronEoverpFilter.eoverpbarrelcut = 1.5
#   replace hltL1NonIsoLargeWindowSingleElectronEoverpFilter.eoverpendcapcut = 2.45
#   replace hltL1NonIsoLargeWindowSingleElectronEoverpFilter.ncandcut = 1
#   replace hltL1NonIsoLargeWindowSingleElectronEoverpFilter.doIsolated = false
#--------------------------------------------------------------------#
# 1/E - 1/P
#--------------------------------------------------------------------#
hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter = copy.deepcopy(hltElectronOneOEMinusOneOPFilter)
import copy
from HLTrigger.Egamma.hltElectronTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter = copy.deepcopy(hltElectronTrackIsolFilter)
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
singleElectronL1NonIsoLargeWindowPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

singleElectronL1NonIsolatedLargeWindow = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedRelaxedSingle")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1NonIsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+cms.SequencePlaceholder("l1NonIsoRecoEcalCandidate")+hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional+hltL1NonIsoLargeWindowSingleElectronEtFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedElectronHcalIsol")+cms.SequencePlaceholder("l1NonIsolatedElectronHcalIsol")+hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter+cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("pixelMatchElectronL1IsoLargeWindowSequenceForHLT")+cms.SequencePlaceholder("pixelMatchElectronL1NonIsoLargeWindowSequenceForHLT")+hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("pixelMatchElectronL1IsoLargeWindowTrackingSequenceForHLT")+cms.SequencePlaceholder("pixelMatchElectronL1NonIsoLargeWindowTrackingSequenceForHLT")+hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter+cms.SequencePlaceholder("l1IsoLargeWindowElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1NonIsoLargeWindowElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoLargeWindowElectronTrackIsol")+cms.SequencePlaceholder("l1NonIsoLargeWindowElectronTrackIsol")+hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter+singleElectronL1NonIsoLargeWindowPresc)
hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional.candNonIsolatedTag = 'l1NonIsoRecoEcalCandidate'
hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional.L1SeedFilterTag = 'l1seedRelaxedSingle'
#     InputTag l1Tag = l1extraParticles:Isolated
hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional.ncandcut = 1
hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional.doIsolated = False
hltL1NonIsoLargeWindowSingleElectronEtFilter.inputTag = 'hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional'
hltL1NonIsoLargeWindowSingleElectronEtFilter.etcut = 18.0
hltL1NonIsoLargeWindowSingleElectronEtFilter.ncandcut = 1
hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter.candTag = 'hltL1NonIsoLargeWindowSingleElectronEtFilter'
hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter.isoTag = 'l1IsolatedElectronHcalIsol'
hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter.nonIsoTag = 'l1NonIsolatedElectronHcalIsol'
hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter.hcalisolbarrelcut = 3.
hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter.hcalisolendcapcut = 3.
hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter.ncandcut = 1
hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter.doIsolated = False
hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter.candTag = 'hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter'
hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter.L1IsoPixelSeedsTag = 'l1IsoLargeWindowElectronPixelSeeds'
hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter.L1NonIsoPixelSeedsTag = 'l1NonIsoLargeWindowElectronPixelSeeds'
hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter.doIsolated = False
#      double   npixelmatchcut         = 1
hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter.ncandcut = 1
hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.candTag = 'hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter'
hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.electronIsolatedProducer = 'pixelMatchElectronsL1IsoLargeWindowForHLT'
hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.electronNonIsolatedProducer = 'pixelMatchElectronsL1NonIsoLargeWindowForHLT'
hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.barrelcut = 999.03
hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.endcapcut = 999.03
hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.ncandcut = 1
hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter.doIsolated = False
#replace  hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter.candTag  = hltL1NonIsoLargeWindowSingleElectronEoverpFilter
hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter.candTag = 'hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter'
hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter.isoTag = 'l1IsoLargeWindowElectronTrackIsol'
hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter.nonIsoTag = 'l1NonIsoLargeWindowElectronTrackIsol'
hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter.pttrackisolcut = 0.06
hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter.ncandcut = 1
hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter.doIsolated = False

