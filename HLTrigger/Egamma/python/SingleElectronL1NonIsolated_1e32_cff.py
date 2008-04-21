# The following comments couldn't be translated into the new config version:

# pixel rechit for SC-pixel matching
# strip rechit for the electron tracking
#hltL1NonIsoSingleElectronEoverpFilter &

#
#
# EgammaHLTRegionalPixelSeedGeneratorProducer

import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.Egamma.hltEgammaL1MatchFilterRegional_cfi import *
#----------- filter modules for the sequence --------------------------------#
#--------------------------------------------------------------------#
# L1 MATCH
#--------------------------------------------------------------------#
hltL1NonIsoSingleElectronL1MatchFilterRegional = copy.deepcopy(hltEgammaL1MatchFilterRegional)
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
hltL1NonIsoSingleElectronEtFilter = copy.deepcopy(hltEgammaEtFilter)
import copy
from HLTrigger.Egamma.hltEgammaHcalIsolFilter_cfi import *
#--------------------------------------------------------------------#
# HCAL ISOL
#--------------------------------------------------------------------#
hltL1NonIsoSingleElectronHcalIsolFilter = copy.deepcopy(hltEgammaHcalIsolFilter)
import copy
from HLTrigger.Egamma.hltElectronPixelMatchFilter_cfi import *
#--------------------------------------------------------------------#
# PIXEL MATCH
#--------------------------------------------------------------------#
hltL1NonIsoSingleElectronPixelMatchFilter = copy.deepcopy(hltElectronPixelMatchFilter)
import copy
from HLTrigger.Egamma.hltElectronOneOEMinusOneOPFilter_cfi import *
# #--------------------------------------------------------------------#
# # E OVER P       
# #--------------------------------------------------------------------#
#  module hltL1NonIsoSingleElectronEoverpFilter = hltElectronEoverpFilter from "HLTrigger/Egamma/data/hltElectronEoverpFilter.cfi"
# 
#  replace hltL1NonIsoSingleElectronEoverpFilter.candTag = hltL1NonIsoSingleElectronPixelMatchFilter
#  replace hltL1NonIsoSingleElectronEoverpFilter.electronIsolatedProducer = pixelMatchElectronsL1IsoForHLT 
#  replace hltL1NonIsoSingleElectronEoverpFilter.electronNonIsolatedProducer = pixelMatchElectronsL1NonIsoForHLT 
#   replace hltL1NonIsoSingleElectronEoverpFilter.eoverpbarrelcut = 1.5
#   replace hltL1NonIsoSingleElectronEoverpFilter.eoverpendcapcut = 2.45
#   replace hltL1NonIsoSingleElectronEoverpFilter.ncandcut = 1
#   replace hltL1NonIsoSingleElectronEoverpFilter.doIsolated = false
#--------------------------------------------------------------------#
# 1/E - 1/P
#--------------------------------------------------------------------#
hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter = copy.deepcopy(hltElectronOneOEMinusOneOPFilter)
import copy
from HLTrigger.Egamma.hltElectronTrackIsolFilter_cfi import *
#--------------------------------------------------------------------#
# TRACK ISOL
#--------------------------------------------------------------------#
hltL1NonIsoSingleElectronTrackIsolFilter = copy.deepcopy(hltElectronTrackIsolFilter)
#--------------------------------------------------------------------#
#--------------------------------------------------------------------#
singleElectronL1NonIsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

singleElectronL1NonIsolated = cms.Sequence(cms.SequencePlaceholder("hltBegin")+cms.SequencePlaceholder("l1seedRelaxedSingle")+cms.SequencePlaceholder("doRegionalEgammaEcal")+cms.SequencePlaceholder("l1IsolatedEcalClusters")+cms.SequencePlaceholder("l1NonIsolatedEcalClusters")+cms.SequencePlaceholder("l1IsoRecoEcalCandidate")+cms.SequencePlaceholder("l1NonIsoRecoEcalCandidate")+hltL1NonIsoSingleElectronL1MatchFilterRegional+hltL1NonIsoSingleElectronEtFilter+cms.SequencePlaceholder("doLocalHcalWithoutHO")+cms.SequencePlaceholder("l1IsolatedElectronHcalIsol")+cms.SequencePlaceholder("l1NonIsolatedElectronHcalIsol")+hltL1NonIsoSingleElectronHcalIsolFilter+cms.SequencePlaceholder("doLocalPixel")+cms.SequencePlaceholder("doLocalStrip")+cms.SequencePlaceholder("pixelMatchElectronL1IsoSequenceForHLT")+cms.SequencePlaceholder("pixelMatchElectronL1NonIsoSequenceForHLT")+hltL1NonIsoSingleElectronPixelMatchFilter+cms.SequencePlaceholder("pixelMatchElectronL1IsoTrackingSequenceForHLT")+cms.SequencePlaceholder("pixelMatchElectronL1NonIsoTrackingSequenceForHLT")+hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter+cms.SequencePlaceholder("l1IsoElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1NonIsoElectronsRegionalRecoTracker")+cms.SequencePlaceholder("l1IsoElectronTrackIsol")+cms.SequencePlaceholder("l1NonIsoElectronTrackIsol")+hltL1NonIsoSingleElectronTrackIsolFilter+singleElectronL1NonIsoPresc)
hltL1NonIsoSingleElectronL1MatchFilterRegional.candIsolatedTag = 'l1IsoRecoEcalCandidate'
hltL1NonIsoSingleElectronL1MatchFilterRegional.candNonIsolatedTag = 'l1NonIsoRecoEcalCandidate'
hltL1NonIsoSingleElectronL1MatchFilterRegional.L1SeedFilterTag = 'l1seedRelaxedSingle'
#     InputTag l1Tag = l1extraParticles:Isolated
hltL1NonIsoSingleElectronL1MatchFilterRegional.ncandcut = 1
hltL1NonIsoSingleElectronL1MatchFilterRegional.doIsolated = False
hltL1NonIsoSingleElectronEtFilter.inputTag = 'hltL1NonIsoSingleElectronL1MatchFilterRegional'
#replace hltL1IsoSingleElectronEtFilter.inputTag = hltL1IsoSingleElectronL1MatchFilter
hltL1NonIsoSingleElectronEtFilter.etcut = 18.0
hltL1NonIsoSingleElectronEtFilter.ncandcut = 1
hltL1NonIsoSingleElectronHcalIsolFilter.candTag = 'hltL1NonIsoSingleElectronEtFilter'
hltL1NonIsoSingleElectronHcalIsolFilter.isoTag = 'l1IsolatedElectronHcalIsol'
hltL1NonIsoSingleElectronHcalIsolFilter.nonIsoTag = 'l1NonIsolatedElectronHcalIsol'
hltL1NonIsoSingleElectronHcalIsolFilter.hcalisolbarrelcut = 3.
hltL1NonIsoSingleElectronHcalIsolFilter.hcalisolendcapcut = 3.
hltL1NonIsoSingleElectronHcalIsolFilter.ncandcut = 1
hltL1NonIsoSingleElectronHcalIsolFilter.doIsolated = False
hltL1NonIsoSingleElectronPixelMatchFilter.candTag = 'hltL1NonIsoSingleElectronHcalIsolFilter'
hltL1NonIsoSingleElectronPixelMatchFilter.L1IsoPixelSeedsTag = 'l1IsoElectronPixelSeeds'
hltL1NonIsoSingleElectronPixelMatchFilter.L1NonIsoPixelSeedsTag = 'l1NonIsoElectronPixelSeeds'
hltL1NonIsoSingleElectronPixelMatchFilter.doIsolated = False
#      double   npixelmatchcut         = 1
hltL1NonIsoSingleElectronPixelMatchFilter.ncandcut = 1
hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter.candTag = 'hltL1NonIsoSingleElectronPixelMatchFilter'
hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter.electronIsolatedProducer = 'pixelMatchElectronsL1IsoForHLT'
hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter.electronNonIsolatedProducer = 'pixelMatchElectronsL1NonIsoForHLT'
hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter.barrelcut = 999.03
hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter.endcapcut = 999.03
hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter.ncandcut = 1
hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter.doIsolated = False
#replace  hltL1NonIsoSingleElectronTrackIsolFilter.candTag  = hltL1NonIsoSingleElectronEoverpFilter
hltL1NonIsoSingleElectronTrackIsolFilter.candTag = 'hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter'
hltL1NonIsoSingleElectronTrackIsolFilter.isoTag = 'l1IsoElectronTrackIsol'
hltL1NonIsoSingleElectronTrackIsolFilter.nonIsoTag = 'l1NonIsoElectronTrackIsol'
hltL1NonIsoSingleElectronTrackIsolFilter.pttrackisolcut = 0.06
hltL1NonIsoSingleElectronTrackIsolFilter.ncandcut = 1
hltL1NonIsoSingleElectronTrackIsolFilter.doIsolated = False

